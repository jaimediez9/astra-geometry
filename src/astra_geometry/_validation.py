"""
_validation.py — Validación compartida de entradas (módulo privado).

Propósito:
    Alojar los helpers comunes de validación de entradas numéricas usados por
    rotation.py y quaternions.py: comprobación de forma, de tipo real, de
    finitud y de la tolerancia atol.

Estado actual:
    Sin implementar. rotation.py y quaternions.py validan sus entradas
    inline, con el mismo patrón repetido y con pequeñas divergencias entre
    módulos (por ejemplo, el manejo de atol y las excepciones capturadas).
    Migrar esos chequeos aquí es trabajo pendiente.

Contrato:
    Módulo privado (prefijo _), no forma parte de la API pública y no debe
    exportarse desde __init__.py. Las convenciones de tolerancia aplicables
    están en docs/conventions.md, sección 7 (Numerical Precision and
    Validation).
"""
