"""
essential.py — Geometría epipolar: matriz esencial y pose relativa.

Alcance (docs/conventions.md, sección 6.2):
    Caso calibrado (intrínsecos K conocidos). Construcción, validación y
    estimación de la matriz esencial E, descomposición en candidatos de pose y
    selección del candidato válido por quiralidad (sección 6.3).
    La matriz fundamental NO está en alcance.

Convención (normativa, sección 6.2):
    x_2^T E x_1 = 0  con  E = [t]_x R,  y (R, t) = T_{2<-1}:
    x_1 son las coordenadas normalizadas en el marco 1, x_2 en el marco 2, y
    x_2 = R x_1 + t. El producto opuesto R [t]_x pertenece al emparejamiento
    invertido y NO es la convención. El frame de t es el marco 2.

Escala y signo:
    E está definida salvo factor escalar no nulo. Por eso solo la DIRECCIÓN de
    t es observable: las poses se devuelven con t̂ unitario.

Justificación matemática de la corrección de signos (revisión §13.3)
--------------------------------------------------------------------
Sea E = [t]_x R con R en SO(3) y t != 0, y sea E = U diag(s, s, 0) V^T su SVD
(los dos valores singulares no nulos son iguales). Entonces

    (a) E x = 0   <=>   x ∝ R^T t        (vector nulo derecho)
    (b) y^T E = 0 <=>   y ∝ t            (vector nulo izquierdo)

porque [t]_x (R x) = 0 <=> R x ∝ t. De ahí u_3 ∝ t y v_3 ∝ R^T t.

Con W = [[0,-1,0],[1,0,0],[0,0,1]], S = diag(1,1,0) y s normalizado a 1:

    [u_3]_x U = +U [e_3]_x   si det(U) = +1   (base derecha)
    [u_3]_x U = -U [e_3]_x   si det(U) = -1
    [e_3]_x W^T = S

(los productos vectoriales de una base ordenada son +u_2, -u_1, 0 cuando la base
es derecha y del signo opuesto cuando es izquierda). Por tanto

    [u_3]_x (U W^T V^T) = det(U) · U S V^T = det(U) · E / s

es decir **E = [t]_x R con R = U W^T V^T y t ∝ det(U) · u_3**. Y como

    det(R) = det(U) · det(W^T) · det(V^T) = det(U) · det(V)

el candidato R = U W^T V^T es una **rotación propia si y solo si
det(U)·det(V) = +1**.

La única libertad de signo admisible es invertir a la vez la tercera columna de U
y de V (s_3 = 0, luego E no cambia), lo que invierte AMBOS determinantes: por
tanto **det(U)·det(V) es un invariante de E**. Si vale -1, ninguna reetiquetación
admisible de (U, V) produce una rotación propia; hay que **refactorizar E como
(-U) diag(s,s,0) V^T** — una SVD válida de -E — con lo que el producto pasa a +1
y la construcción anterior aplica.

Sin esa corrección el resultado depende de la rama que elija la SVD y puede salir
una reflexión (det = -1): no basta con normalizar signos a posteriori, porque el
producto es invariante.

Este módulo importa camera.py, rotation.py, transforms.py y triangulation.py,
y nunca al contrario.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .camera import _as_intrinsic
from .rotation import is_rotation_matrix
from .transforms import _as_translation, transformation_matrix
from .triangulation import triangulate

__all__ = [
    "decompose_essential_matrix",
    "essential_matrix",
    "estimate_essential_matrix",
    "estimate_relative_pose",
    "is_essential_matrix",
    "select_candidate",
]

_W = np.array([
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])

# Correspondencias mínimas del método lineal de 8 puntos.
_MIN_CORRESPONDENCES = 8

_SQRT2 = float(np.sqrt(2.0))


def is_essential_matrix(E: ArrayLike, *, atol: float = 1e-8) -> bool:
    """Comprueba si E es una matriz esencial válida.

    Condiciones: forma (3, 3), real y finita; rango 2 y dos valores singulares
    iguales, es decir espectro (s, s, 0) con s > 0.

    Como E está definida salvo escala (sección 6.2), la comprobación es
    necesariamente invariante de escala: el espectro se normaliza por su mayor
    valor singular y la tolerancia se aplica de forma ABSOLUTA sobre ese
    espectro normalizado (rtol implícito 0, sección 7.2). Una tolerancia
    absoluta sobre las entradas de E no tendría sentido, porque esas entradas no
    están fijadas por la geometría.

    Devuelve False ante entradas inválidas en lugar de lanzar excepción.
    """
    try:
        E = np.asarray(E)
        atol = float(atol)
    except (TypeError, ValueError, OverflowError):
        return False

    if E.shape != (3, 3):
        return False

    if np.iscomplexobj(E) or not np.issubdtype(E.dtype, np.number):
        return False

    if not np.isfinite(atol) or atol < 0.0:
        return False

    try:
        E = E.astype(np.float64)

        if not np.all(np.isfinite(E)):
            return False

        singular_values = np.linalg.svd(E, compute_uv=False)

    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
        return False

    if not np.all(np.isfinite(singular_values)):
        return False

    if singular_values[0] <= 0.0:
        return False

    spectrum = singular_values / singular_values[0]

    return bool(abs(spectrum[0] - spectrum[1]) <= atol and spectrum[2] <= atol)


def essential_matrix(
    R: ArrayLike,
    t: ArrayLike,
    *,
    atol: float = 1e-8,
) -> NDArray[np.float64]:
    """Construye E = [t]_x R a partir de la pose relativa T_{2<-1}.

    La orientación es la de la sección 6.2 y NO es intercambiable: E = [t]_x R.
    El resultado se devuelve sin normalizar, de modo que su escala es ||t||
    (E está definida salvo escala de todos modos).

    Args:
        R: Matriz de rotación (3, 3) de T_{2<-1}.
        t: Traslación (3,) de T_{2<-1}, expresada en el marco 2. No puede ser
            nula: con t = 0 la matriz sería la nula y una rotación pura no
            determina la geometría epipolar (sección 6.2).
        atol: Tolerancia absoluta para validar R (keyword-only).

    Returns:
        Matriz esencial float64 de forma (3, 3).

    Raises:
        ValueError: Si R no es una rotación de SO(3), si t no es un vector real
            finito de forma (3,), o si t es nulo.
    """
    if np.iscomplexobj(np.asarray(R)):
        raise ValueError("R debe representar una matriz de rotación válida de SO(3)")

    if not is_rotation_matrix(R, atol=atol):
        raise ValueError("R debe representar una matriz de rotación válida de SO(3)")

    R = np.asarray(R, dtype=np.float64)
    t = _as_translation(t, "t")

    if np.linalg.norm(t) == 0.0:
        raise ValueError(
            "t no puede ser nulo: E sería la matriz nula (rotación pura, "
            "geometría epipolar indeterminada)"
        )

    skew = np.array([
        [0.0, -t[2], t[1]],
        [t[2], 0.0, -t[0]],
        [-t[1], t[0], 0.0],
    ])

    return skew @ R


def estimate_essential_matrix(
    x1: ArrayLike,
    x2: ArrayLike,
    *,
    atol: float = 1e-8,
) -> NDArray[np.float64]:
    """Estima E por el método lineal de 8 puntos, normalizado (Hartley).

    Entradas en COORDENADAS NORMALIZADAS (x̂ = K^{-1}[u, v, 1]^T), que es el caso
    calibrado en alcance. Procedimiento:

    1. Normalización de Hartley de cada conjunto (centroide al origen y escala
       tal que la distancia media sea sqrt(2)), que mejora el condicionamiento.
    2. Sistema lineal A e = 0, una fila por correspondencia, resuelto por SVD.
    3. Desnormalización.
    4. Proyección a la variedad de las matrices esenciales: se anula el tercer
       valor singular (mejor aproximación de rango 2 en norma de Frobenius,
       teorema de Eckart-Young-Mirsky) y se igualan los dos supervivientes
       (condición propia de la variedad esencial). El orden importa: la
       desnormalización es una congruencia con matrices no ortogonales y
       rompería la estructura, así que la proyección va DESPUÉS.
    5. Normalización canónica de escala a sigma = 1 (||E||_F = sqrt(2)), una
       elección arbitraria pero reproducible.

    Args:
        x1: Coordenadas normalizadas (N, 2) en la cámara 1.
        x2: Coordenadas normalizadas (N, 2) en la cámara 2.
        atol: Tolerancia absoluta no negativa (keyword-only).

    Returns:
        Matriz esencial float64 de forma (3, 3) con ||E||_F = sqrt(2).

    Raises:
        ValueError: Si las entradas no son (N, 2) reales finitas, si los tamaños
            no coinciden, si hay menos de 8 correspondencias, o si la solución
            es degenerada (E se anula).
    """
    x1 = _as_image_points(x1, "x1")
    x2 = _as_image_points(x2, "x2")
    _as_atol(atol)

    if x1.shape[0] != x2.shape[0]:
        raise ValueError(
            "x1 y x2 deben tener el mismo número de correspondencias "
            f"({x1.shape[0]} frente a {x2.shape[0]})"
        )

    if x1.shape[0] < _MIN_CORRESPONDENCES:
        raise ValueError(
            f"el método de 8 puntos necesita al menos {_MIN_CORRESPONDENCES} "
            f"correspondencias ({x1.shape[0]} recibidas)"
        )

    normal1, homogeneous1 = _hartley_normalize(x1)
    normal2, homogeneous2 = _hartley_normalize(x2)

    rows = np.einsum("ni,nj->nij", homogeneous2, homogeneous1).reshape(-1, 9)

    # Solo el caso "corto" (N < 9) necesita la V completa: con una matriz 8x9 la
    # SVD reducida devuelve una Vh de (8, 9), que contiene los ocho primeros
    # vectores singulares derechos y NO el noveno, que es precisamente el del
    # núcleo y por tanto la solución. Medido sobre el sistema real: la reducida
    # da un residuo |A v|/|A| de 2e-2 frente a 4e-17 con la completa, es decir un
    # resultado erróneo SILENCIOSO, no una excepción.
    #
    # Para N >= 9 se cumple min(N, 9) = 9, así que la reducida ya devuelve Vh de
    # (9, 9) — bit-idéntica a la completa en este build — y evita materializar la
    # U de (N, N), que es lo que hacía el coste O(N^2) de memoria y tiempo: a
    # N = 2e4 la completa pide 4.1 GB y 77 s; la reducida, 57 MB y 5 ms.
    _, _, Vt = np.linalg.svd(rows, full_matrices=rows.shape[0] < rows.shape[1])
    E_solution = Vt[-1].reshape(3, 3)

    if not np.all(np.isfinite(E_solution)):
        raise ValueError("la estimación de E no es finita (entradas degeneradas)")

    E_raw = normal2.T @ E_solution @ normal1

    # Proyección a la variedad esencial (después de desnormalizar).
    U, singular, Vt_E = np.linalg.svd(E_raw)
    sigma = 0.5 * (singular[0] + singular[1])

    if sigma <= 0.0:
        raise ValueError(
            "estimación degenerada: todas las correspondencias son "
            "consistentes con la matriz nula (sin información epipolar)"
        )

    E = U @ np.diag([sigma, sigma, 0.0]) @ Vt_E

    return E / (np.linalg.norm(E) / _SQRT2)


def decompose_essential_matrix(
    E: ArrayLike,
    *,
    atol: float = 1e-8,
) -> tuple[tuple[NDArray[np.float64], NDArray[np.float64]], ...]:
    """Descompone E en los cuatro candidatos de pose (R, t̂) de T_{2<-1}.

    Los candidatos son {R_1, R_2} x {+t̂, -t̂}, donde R_2 es la pareja "torcida"
    de R_1. Los cuatro satisfacen la restricción epipolar; el válido se
    selecciona por quiralidad con select_candidate.

    La corrección de signos (det(U)·det(V) < 0 => refactorizar E como -E) está
    justificada en el docstring del módulo.

    Args:
        E: Matriz esencial (3, 3), validada con is_essential_matrix.
        atol: Tolerancia absoluta para validar E (keyword-only).

    Returns:
        Tupla de 4 pares (R, t̂): R en SO(3) de forma (3, 3) y t̂ unitario (3,).
        Orden nominal: (R_1, +u3), (R_1, -u3), (R_2, +u3), (R_2, -u3).

        El **orden no es una garantía estable**: como los dos valores singulares
        no nulos de E son iguales, los vectores singulares del subespacio 2-D no
        son únicos y la librería de SVD puede repartir esa base de otra forma
        (incluso ante un simple reescalado de E). Lo que SÍ es invariante es el
        **conjunto** de los cuatro candidatos, que son todas las soluciones de
        E = [t]_x R. Consumir la salida como conjunto: select_candidate puntúa
        los cuatro y por eso es inmune al orden.

    Raises:
        ValueError: Si E no es una matriz esencial válida.
    """
    atol_value = _as_atol(atol)
    E = _as_essential(E, atol_value)

    U, _, Vt = np.linalg.svd(E)
    V = Vt.T

    if np.linalg.det(U) * np.linalg.det(V) < 0.0:
        U, _, Vt = np.linalg.svd(-E)
        V = Vt.T

    # Con sigma_3 = 0 se puede invertir a la vez u_3 y v_3 sin cambiar E, lo que
    # fija det(U) = det(V) = +1: la hipótesis de la demostración del módulo.
    #
    # NOTA: este paso NO es funcionalmente crítico y no debe confundirse con la
    # corrección del bloque anterior. Invertir u_3 y v_3 no altera ni R_proper ni
    # R_twisted (porque D W^T D = W^T con D = diag(1,1,-1)) ni el CONJUNTO de
    # candidatos: solo fija qué signo de t̂ queda emparejado con cada rotación en
    # el orden nominal. Se conserva para que la salida sea coherente con el orden
    # que documenta esta función; el consumidor correcto es select_candidate, que
    # puntúa los cuatro candidatos y es inmune al orden.
    if np.linalg.det(U) < 0.0:
        U = U.copy()
        V = V.copy()
        U[:, 2] *= -1.0
        V[:, 2] *= -1.0
        Vt = V.T

    R_proper = U @ _W.T @ Vt      # E = [u_3]_x R_proper, con t ∝ +u_3
    R_twisted = U @ _W @ Vt       # pareja torcida
    t_hat = U[:, 2].copy()

    return (
        (R_proper, t_hat.copy()),
        (R_proper, -t_hat.copy()),
        (R_twisted, t_hat.copy()),
        (R_twisted, -t_hat.copy()),
    )


def select_candidate(
    candidates,
    pixels1: ArrayLike,
    pixels2: ArrayLike,
    K1: ArrayLike,
    K2: ArrayLike,
    *,
    atol: float = 1e-8,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Selecciona el candidato válido por quiralidad (profundidad positiva).

    Triangula las correspondencias con cada candidato y cuenta cuántas quedan
    con profundidad positiva en AMBAS cámaras (sección 6.3). Gana el candidato
    con más puntos delante, y debe haber un ganador único.

    Args:
        candidates: Secuencia de pares (R, t) como los de
            decompose_essential_matrix. La traslación se normaliza a unitaria.
        pixels1: Observaciones (N, 2) en píxeles en la cámara 1.
        pixels2: Observaciones (N, 2) en píxeles en la cámara 2.
        K1: Intrínsecos (3, 3) de la cámara 1.
        K2: Intrínsecos (3, 3) de la cámara 2.
        atol: Tolerancia absoluta no negativa (keyword-only).

    Returns:
        El par (R, t̂) seleccionado, con t̂ unitario.

    Raises:
        ValueError: Si las entradas no cumplen su contrato, si algún candidato
            no es una pose válida, o si no hay un ganador único.
    """
    _as_atol(atol)
    _as_intrinsic(K1, "K1")
    _as_intrinsic(K2, "K2")
    _as_image_points(pixels1, "pixels1")
    _as_image_points(pixels2, "pixels2")

    checked = [_as_candidate(R, t) for (R, t) in candidates]

    if not checked:
        raise ValueError("se necesita al menos un candidato de pose")

    counts: list[int] = []
    for R, t_hat in checked:
        pose = transformation_matrix(R, t_hat)
        try:
            points = triangulate(pixels1, pixels2, K1, K2, pose, atol=atol)
        except ValueError:
            counts.append(0)  # triangulación degenerada con este candidato
            continue

        depth1 = points[:, 2]
        depth2 = (points @ R.T + t_hat)[:, 2]
        counts.append(int(np.count_nonzero((depth1 > 0.0) & (depth2 > 0.0))))

    best = int(np.argmax(counts))

    if counts[best] == 0:
        raise ValueError(
            "ningún candidato deja puntos con profundidad positiva en ambas "
            "cámaras: configuración degenerada o poses inválidas"
        )

    if counts.count(counts[best]) > 1:
        raise ValueError(
            f"quiralidad ambigua: {counts.count(counts[best])} candidatos "
            f"empatan con {counts[best]} puntos delante"
        )

    return checked[best]


def estimate_relative_pose(
    pixels1: ArrayLike,
    pixels2: ArrayLike,
    K1: ArrayLike,
    K2: ArrayLike,
    *,
    atol: float = 1e-8,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estima la pose relativa T_{2<-1} desde correspondencias en píxeles.

    Encadena estimación de E (8 puntos normalizado) -> descomposición ->
    selección por quiralidad.

    Returns:
        El par (R, t̂): la rotación y la DIRECCIÓN de la traslación con el signo
        resuelto por quiralidad, con t̂ unitario. La magnitud de t no es
        observable desde E (sección 6.2).

    Raises:
        ValueError: Bajo las mismas condiciones que las funciones que encadena.
    """
    K1 = _as_intrinsic(K1, "K1")
    K2 = _as_intrinsic(K2, "K2")
    p1 = _as_image_points(pixels1, "pixels1")
    p2 = _as_image_points(pixels2, "pixels2")

    x1 = _normalize(p1, K1)
    x2 = _normalize(p2, K2)

    E = estimate_essential_matrix(x1, x2, atol=atol)
    candidates = decompose_essential_matrix(E, atol=atol)

    return select_candidate(candidates, p1, p2, K1, K2, atol=atol)


# --- helpers privados -------------------------------------------------------


def _normalize(
    pixels: NDArray[np.float64],
    K: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Coordenadas de píxel (N, 2) a coordenadas normalizadas (N, 2)."""
    homogeneous = np.hstack((pixels, np.ones((pixels.shape[0], 1))))
    normalized = homogeneous @ np.linalg.inv(K).T
    return normalized[:, :2] / normalized[:, 2:3]


def _hartley_normalize(points: NDArray[np.float64]):
    """Normalización de Hartley de puntos (N, 2).

    Devuelve (T, homogéneos) con T (3, 3) tal que x_n = T x_h, y los puntos ya
    normalizados en forma homogénea (N, 3), que es la que consume el sistema
    lineal de 8 puntos.

    Raises:
        ValueError: Si todos los puntos coinciden y la escala no está definida.
    """
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_distance = np.linalg.norm(shifted, axis=1).mean()

    if mean_distance == 0.0:
        raise ValueError(
            "no se puede normalizar: todas las correspondencias coinciden "
            "(configuración degenerada)"
        )

    scale = _SQRT2 / mean_distance
    normalized = shifted * scale

    T = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0],
    ])
    homogeneous = np.hstack((normalized, np.ones((points.shape[0], 1))))

    return T, homogeneous


def _as_image_points(points: ArrayLike, nombre: str) -> NDArray[np.float64]:
    """Convierte points en un array float64 de forma (N, 2) validado.

    Helper privado (validación que se trasladará a _validation.py).

    Raises:
        ValueError: Si points no es un conjunto real finito de forma (N, 2).
    """
    try:
        points = np.asarray(points)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{nombre} debe ser convertible a un array real float64"
        ) from exc

    if np.iscomplexobj(points):
        raise ValueError(f"{nombre} debe contener valores reales")

    try:
        points = points.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{nombre} debe ser convertible a un array real float64"
        ) from exc

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{nombre} debe tener forma (N, 2)")

    if not np.all(np.isfinite(points)):
        raise ValueError(f"{nombre} debe contener valores finitos")

    return points


def _as_essential(E: ArrayLike, atol: float) -> NDArray[np.float64]:
    """Convierte E en un array float64 (3, 3) validado como matriz esencial.

    Raises:
        ValueError: Si E no es una matriz esencial válida.
    """
    try:
        E = np.asarray(E)
    except (TypeError, ValueError) as exc:
        raise ValueError("E debe ser convertible a una matriz real float64") from exc

    if np.iscomplexobj(E):
        raise ValueError("E debe contener valores reales")

    try:
        E = E.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("E debe ser convertible a una matriz real float64") from exc

    if E.shape != (3, 3):
        raise ValueError("E debe tener forma (3, 3)")

    if not np.all(np.isfinite(E)):
        raise ValueError("E debe contener valores finitos")

    if not is_essential_matrix(E, atol=atol):
        raise ValueError(
            "E debe ser una matriz esencial válida (rango 2 y dos valores "
            "singulares iguales)"
        )

    return E


def _as_candidate(R: ArrayLike, t: ArrayLike):
    """Valida un candidato (R, t) y devuelve (R, t unitario).

    Raises:
        ValueError: Si R no es una rotación o t no es un vector (3,) finito no nulo.
    """
    if np.iscomplexobj(np.asarray(R)):
        raise ValueError("cada candidato debe incluir una rotación válida de SO(3)")

    if not is_rotation_matrix(R):
        raise ValueError("cada candidato debe incluir una rotación válida de SO(3)")

    t = _as_translation(t, "t")
    norm = np.linalg.norm(t)

    if norm == 0.0:
        raise ValueError("cada candidato debe incluir una traslación no nula")

    return np.asarray(R, dtype=np.float64), t / norm


def _as_atol(atol) -> float:
    """Convierte atol en un float real finito no negativo.

    Raises:
        ValueError: Si atol no es un escalar real finito y no negativo.
    """
    try:
        atol = np.asarray(atol)

        if atol.shape != () or np.iscomplexobj(atol):
            raise ValueError("atol debe ser un escalar real finito no negativo")

        atol = float(atol)

    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "atol debe ser un escalar real finito no negativo"
        ) from exc

    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol debe ser un escalar real finito no negativo")

    return atol
