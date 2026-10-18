# Astra Geometry — Mathematical Conventions

**Status:** Initial specification

**Scope:** Coordinate systems, rotations, rigid transformations, numerical precision, and camera geometry.

## 1. Coordinate Systems

* All coordinate systems are three-dimensional, Cartesian, and right-handed.
* Vectors are represented as column vectors in mathematical notation.
* Lengths use meters by convention.
* Angles are expressed in radians.
* Coordinate frames must have explicit identifiers and documented relationships.

## 2. Rotation Matrices — $\mathrm{SO}(3)$

A rotation matrix $R \in \mathrm{SO}(3)$ satisfies:

$$
R^\mathsf{T}R = I_3, \qquad \det(R) = +1
$$

- Rotation matrices are represented as $3\times3$ matrices.
- Matrices act on column vectors by left multiplication: $p' = Rp$.

## 3. Rigid Transformations — $\mathrm{SE}(3)$

A rigid transformation is represented by:

$$
T = \begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix}
$$

where $R \in \mathrm{SO}(3)$ and $t \in \mathbb{R}^3$.

Homogeneous points:

$$
\tilde{p} = \begin{bmatrix}
p \\
1
\end{bmatrix}
$$

The transformation acts as:

$$
\tilde{p}' = T\tilde{p}
$$

## 4. Inverse Transformation

For $T$:

$$
T^{-1} = \begin{bmatrix}
R^\mathsf{T} & -R^\mathsf{T}t \\
0 & 1
\end{bmatrix}
$$

## 6. Camera Geometry

The camera intrinsic matrix is:

$$
K = \begin{bmatrix}
f_x & s & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

Projection equation:

$$
\lambda \begin{bmatrix}
u \\
v \\
1
\end{bmatrix} = K \begin{bmatrix}
X_c \\
Y_c \\
Z_c
\end{bmatrix}
$$

where $\lambda$ is the projective scale.

## 7. Numerical Precision and Validation

* The default floating-point representation for the initial C++ implementation is `double`.
* Numerical comparisons must use explicit tolerances appropriate to the operation.
* Absolute and relative tolerances must be distinguished.
* Non-finite values must be handled explicitly.
* Invalid dimensions and degenerate geometric configurations must be rejected or reported according to the API contract.
* Validation must distinguish exact mathematical properties from approximate floating-point checks.

## 8. Testing Requirements

Core operations must be tested against mathematical properties, not solely against example outputs.

Required properties include:

* Rotation orthogonality and determinant.
* Preservation of distances under rigid transformations.
* Composition with the identity transformation.
* Transformation-inverse consistency.
* Quaternion normalization and equivalence of $q$ and $-q$.
* Camera projection and reprojection consistency.
* Recovery of known rigid transformations from synthetic correspondences.

## 9. Scope and Guarantees

These conventions define the intended mathematical behavior of Astra Geometry. They do not, by themselves, establish implementation correctness, numerical conditioning, execution-time bounds, or hard real-time suitability. Such properties require separate tests and analysis.
