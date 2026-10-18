# Astra Geometry — Mathematical Conventions

**Status:** Initial specification  
**Scope:** Coordinate systems, rotations, rigid transformations, numerical precision, and camera geometry.

## 1. Coordinate Systems

- All coordinate systems are three-dimensional, Cartesian, and right-handed.
- Vectors are represented as column vectors in mathematical notation.
- Lengths use meters by convention.
- Angles are expressed in radians.
- Coordinate frames must have explicit identifiers and documented relationships.

## 2. Rotation Matrices — SO(3)

A rotation matrix \(R \in SO(3)\) satisfies:

$$
R^\mathsf{T}R = I_3,\qquad \det(R)=+1
$$

- Rotation matrices are represented as \(3\times3\) matrices.
- Matrices act on column vectors by left multiplication: \(p' = Rp\).
- Rotation matrices must preserve Euclidean lengths and orientation.
- Approximate validity is evaluated using explicit numerical tolerances.
- Reflections and non-orthogonal matrices are not valid rotations.

## 3. Rigid Transformations — SE(3)

A rigid transformation is represented by:

$$
T =
\begin{bmatrix}
R & t\\
0 & 1
\end{bmatrix}
$$

where \(R\in SO(3)\) and \(t\in\mathbb{R}^3\).

Homogeneous points are represented as:

$$
\tilde p =
\begin{bmatrix}
p\\1
\end{bmatrix}
$$

The transformation acts as:

$$
\tilde p' = T\tilde p
$$

The translation vector and the point coordinates must use the same length unit.

## 4. Transformation Direction and Composition

The notation \(T_{A\leftarrow B}\) denotes a transformation that converts coordinates expressed in frame \(B\) into coordinates expressed in frame \(A\).

Thus:

$$
p_A = T_{A\leftarrow B}p_B
$$

Composition follows matrix multiplication:

$$
T_{A\leftarrow C}
=
T_{A\leftarrow B}T_{B\leftarrow C}
$$

The rightmost transformation acts first.

The inverse satisfies:

$$
T_{A\leftarrow B}^{-1}=T_{B\leftarrow A}
$$

For \(T=[R\;t;0\;1]\):

$$
T^{-1} =
\begin{bmatrix}
R^\mathsf{T} & -R^\mathsf{T}t\\
0 & 1
\end{bmatrix}
$$

## 5. Quaternions

- Quaternions use the Hamilton convention.
- Components are stored in the order \((w,x,y,z)\), with the scalar component first.
- Unit quaternions represent rotations.
- The quaternions \(q\) and \(-q\) represent the same physical rotation.
- Composition order must agree with the rotation-matrix convention.
- Conversion routines must document normalization and numerical edge-case behavior.

## 6. Camera Geometry

The initial camera model is the pinhole model.

- Camera coordinates use a right-handed frame.
- The positive \(Z\)-axis is the optical direction.
- The camera intrinsic matrix is:

$$
K =
\begin{bmatrix}
f_x & s & c_x\\
0 & f_y & c_y\\
0 & 0 & 1
\end{bmatrix}
$$

- Camera extrinsics use the same transformation-direction convention defined in Section 4.
- Projection is defined for points with positive camera-frame depth \(Z_c>0\).
- Lens distortion is outside the initial pinhole model and must be specified separately if introduced.

For a point expressed in camera coordinates:

$$
\lambda
\begin{bmatrix}
u\\v\\1
\end{bmatrix}
=
K
\begin{bmatrix}
X_c\\Y_c\\Z_c
\end{bmatrix}
$$

where \(\lambda\) is the projective scale.

## 7. Numerical Precision and Validation

- The default floating-point representation for the initial C++ implementation is `double`.
- Numerical comparisons must use explicit tolerances appropriate to the operation.
- Absolute and relative tolerances must be distinguished.
- Non-finite values must be handled explicitly.
- Invalid dimensions and degenerate geometric configurations must be rejected or reported according to the API contract.
- Validation must distinguish exact mathematical properties from approximate floating-point checks.

## 8. Testing Requirements

Core operations must be tested against mathematical properties, not solely against example outputs.

Required properties include:

- Rotation orthogonality and determinant.
- Preservation of distances under rigid transformations.
- Composition with the identity transformation.
- Transformation-inverse consistency.
- Quaternion normalization and equivalence of \(q\) and \(-q\).
- Camera projection and reprojection consistency.
- Recovery of known rigid transformations from synthetic correspondences.

## 9. Scope and Guarantees

These conventions define the intended mathematical behavior of Astra Geometry. They do not, by themselves, establish implementation correctness, numerical conditioning, execution-time bounds, or hard real-time suitability. Such properties require separate tests and analysis.
