import os
import cv2 as cv
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_session(model_path: str, prefer_gpu: bool = True) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ort.get_available_providers()
    if prefer_gpu and "CUDAExecutionProvider" in providers:
        ep = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        ep = ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, sess_options=so, providers=ep)

def get_model_io_shape(session: ort.InferenceSession):
    inp = session.get_inputs()[0]
    shape = inp.shape  # [N,C,H,W]
    H = shape[-2] if isinstance(shape[-2], int) else None
    W = shape[-1] if isinstance(shape[-1], int) else None
    return H, W

def choose_infer_size(orig_h, orig_w, fixed_hw=None, target_long=2048, align=32):
    # Если у модели фиксированный вход — вернём его
    if fixed_hw is not None:
        return fixed_hw
    # Иначе выберем разумное разрешение одного прогона
    long_side = max(orig_h, orig_w)
    scale = min(1.0, target_long / float(long_side))
    h = int(round(orig_h * scale / align) * align)
    w = int(round(orig_w * scale / align) * align)
    h = max(align, h)
    w = max(align, w)
    return (h, w)

def preprocess_bgr(image_bgr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    # Resize с сохранением пропорций уже учтён в choose_infer_size
    x = cv.resize(image_bgr, (out_w, out_h), interpolation=cv.INTER_AREA)
    x = cv.cvtColor(x, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)  # CHW
    x = x[None, ...].astype(np.float32)  # NCHW
    return x

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def run_model(session: ort.InferenceSession, x_nchw: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: x_nchw})[0]
    return out  # ожидаем (N, C, H, W) или (N, H, W)

def to_prob_map(out: np.ndarray, binary=True) -> np.ndarray:
    # Приводим к (H,W) для бинарной или (H,W,C) для мультикласса в [0,1]
    if out.ndim == 3:
        out = out[0]  # (C,H,W) или (H,W)
    elif out.ndim == 4:
        out = out[0]  # (C,H,W)
    else:
        raise RuntimeError(f"Unexpected output shape: {out.shape}")

    if out.ndim == 2:
        # (H,W) — уже вероятности или логиты
        if out.min() < 0 or out.max() > 1:
            out = sigmoid(out)
        prob = out.astype(np.float32)
        prob = np.clip(prob, 0.0, 1.0)
        return prob  # (H,W)
    else:
        # (C,H,W)
        C, H, W = out.shape
        if C == 1 and binary:
            ch = out[0]
            if ch.min() < 0 or ch.max() > 1:
                ch = sigmoid(ch)
            prob = np.clip(ch.astype(np.float32), 0, 1)
            return prob  # (H,W)
        else:
            sm = softmax(out[None, ...], axis=1)[0]  # (C,H,W)
            sm = np.transpose(sm, (1, 2, 0)).astype(np.float32)  # (H,W,C)
            return sm

def edge_aware_refine(prob_hr0: np.ndarray, guide_bgr: np.ndarray,
                      method: str = "guided",
                      r: int = 32, eps: float = 1e-3,
                      dt_sigma_spatial: float = 32.0, dt_sigma_color: float = 0.01,
                      jbf_d: int = 9, jbf_sigma_color: float = 0.1, jbf_sigma_space: float = 25.0) -> np.ndarray:
    """
    prob_hr0: (H,W) или (H,W,C) float32 в [0,1] — уже ресайзнутая до исходного размера карта вероятностей.
    guide_bgr: исходное изображение (H,W,3) BGR.
    method: "guided" | "dt" | "jbf"
    """
    H, W = guide_bgr.shape[:2]
    prob = prob_hr0.copy().astype(np.float32)
    guide = guide_bgr

    # Пытаемся использовать ximgproc; если нет — fallback
    has_xi = hasattr(cv, "ximgproc")

    def _clip01(x):
        return np.clip(x, 0.0, 1.0)

    if prob.ndim == 2:
        prob_ch = [prob]
    else:
        prob_ch = [prob[..., c] for c in range(prob.shape[2])]

    refined = []

    if method == "guided" and has_xi:
        # Guided filter (быстро и качественно для границ)
        for ch in prob_ch:
            rf = cv.ximgproc.guidedFilter(guide=guide, src=ch, radius=int(r), eps=float(eps))
            refined.append(_clip01(rf))
    elif method == "dt" and has_xi:
        # Domain Transform — ещё быстрее на больших картинках, хорошо держит границы
        for ch in prob_ch:
            rf = cv.ximgproc.dtFilter(guide=guide, src=ch,
                                      sigmaSpatial=float(dt_sigma_spatial),
                                      sigmaColor=float(dt_sigma_color))
            refined.append(_clip01(rf))
    elif method == "jbf" and has_xi:
        # Joint bilateral (чуть медленнее, но тоже ок)
        for ch in prob_ch:
            rf = cv.ximgproc.jointBilateralFilter(guide=guide, src=ch, d=int(jbf_d),
                                                  sigmaColor=float(jbf_sigma_color*255.0),
                                                  sigmaSpace=float(jbf_sigma_space))
            refined.append(_clip01(rf))
    else:
        # Fallback: обычный bilateral по вероятностям (без «joint»), как запасной вариант
        # Лучше установить opencv-contrib-python для ximgproc.
        for ch in prob_ch:
            rf = cv.bilateralFilter(ch, d=9, sigmaColor=50, sigmaSpace=50)
            refined.append(_clip01(rf))

    if prob.ndim == 2:
        return refined[0]
    else:
        out = np.stack(refined, axis=-1)
        # Нормализуем, чтобы суммы вероятностей по классам были 1
        s = np.sum(out, axis=-1, keepdims=True) + 1e-8
        return out / s

def upscale_and_refine(original_bgr: np.ndarray,
                       prob_lr: np.ndarray,
                       method: str,
                       base_r: int,
                       eps: float) -> np.ndarray:
    H, W = original_bgr.shape[:2]
    # Базовый апскейл до оригинала
    if prob_lr.ndim == 2:
        prob_hr0 = cv.resize(prob_lr, (W, H), interpolation=cv.INTER_LINEAR)
    else:
        C = prob_lr.shape[2]
        prob_hr0 = cv.resize(prob_lr, (W, H), interpolation=cv.INTER_LINEAR)
        prob_hr0 = np.clip(prob_hr0, 0.0, 1.0)

    # Радиус подбираем пропорционально масштабу апскейла
    # Чем сильнее уменьшали — тем больше радиус
    # Пример: если long side инференса 2048 для оригинала 8192 => scale ~4 => r ~ base_r*scale
    # Оценим scale как отношение диагоналей
    # Возьмём отношение по ширине (сойдёт):
    scale = W / max(1, prob_lr.shape[1])
    r = max(8, int(base_r * scale))

    refined = edge_aware_refine(prob_hr0, original_bgr, method=method, r=r, eps=eps)
    return refined

def segment_with_smart_upscale(model_path: str,
                               image_path: str,
                               out_mask_path: str = "output_mask.png",
                               prefer_gpu: bool = True,
                               # Инференс без тайлов
                               target_long_side: int = 2048,
                               # Edge-aware upsampling
                               refine_method: str = "guided",  # "guided" | "dt" | "jbf"
                               base_radius: int = 8,
                               gf_eps: float = 1e-3,
                               binary: bool = True,
                               threshold: float = 0.5,
                               target_class: int = 1
                               ):
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    assert img is not None, f"Не удалось открыть {image_path}"
    H0, W0 = img.shape[:2]

    session = load_session(model_path, prefer_gpu=prefer_gpu)
    inH, inW = get_model_io_shape(session)
    fixed = (inH, inW) if (inH is not None and inW is not None) else None


    infer_h, infer_w = choose_infer_size(H0, W0, fixed_hw=fixed, target_long=target_long_side, align=32)

    x = preprocess_bgr(img, infer_h, infer_w)

    out = run_model(session, x)
    prob_lr = to_prob_map(out, binary=binary)

    if not binary and prob_lr.ndim == 3:
        if target_class is None:
            # argmax
            label_lr = np.argmax(prob_lr, axis=2).astype(np.uint8)  # (h,w)
            # вероятности выбранного класса для сглаживания границ
            prob_lr = (label_lr == label_lr).astype(np.float32)
        else:
            prob_lr = prob_lr[..., int(target_class)]

    # Апскейл + edge-aware refine
    prob_hr = upscale_and_refine(img, prob_lr, method=refine_method, base_r=base_radius, eps=gf_eps)

    mask = (prob_hr >= float(threshold)).astype(np.uint8) * 255
    cv.imwrite(out_mask_path, mask)

    overlay = img.copy()
    red = np.zeros_like(img); red[:, :, 2] = 255
    blended = cv.addWeighted(img, 0.6, red, 0.4, 0.0)
    overlay[mask > 0] = blended[mask > 0]
    cv.imwrite(os.path.splitext(out_mask_path)[0] + "_overlay.jpg", overlay)

if __name__ == "__main__":
    segment_with_smart_upscale(
        model_path="skyseg.onnx",
        image_path="eval/4.jpg",
        out_mask_path="output_mask.png",
        prefer_gpu=True,
        target_long_side=2048,   # можно 1536/2048/3072
        refine_method="guided",  # "guided", "dt" (ещё быстрее), "jbf"
        base_radius=8,
        gf_eps=1e-3,
        binary=True,
        threshold=0.5
    )