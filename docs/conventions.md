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

Rotation matrices are represented as $3\times3$ matrices.

Matrices act on column vectors by left multiplication: $p' = Rp$.

The inverse of a valid rotation matrix is its transpose: $R^{-1}=R^\mathsf{T}$.

The composition $R=R_2R_1$ applies $R_1$ first and $R_2$ second.

Composition is associative but not generally commutative: $R_3(R_2R_1)=(R_3R_2)R_1$, while $R_2R_1\ne R_1R_2$ in general.

The identity matrix $I_3$ represents the identity rotation.

### 2.1 NumPy Array Convention

The mathematical convention uses column vectors. The initial NumPy implementation stores point sets as arrays of shape $(N,3)$, with one point per row.

For a point set $P$, the equivalent operation is:

$$
P'=PR^\mathsf{T}
$$

This is an implementation-level storage convention; it does not change the underlying mathematical convention.

## 3. Quaternion Convention

Quaternions representing rotations use the Hamilton convention and are stored in scalar-first order:


$$
q=(w,x,y,z)=(w,\mathbf v)
$$


where $w$ is the scalar component and $\mathbf v=(x,y,z)$ is the vector component.

Unit quaternions represent rotations in three dimensions. The quaternions $q$ and $-q$ represent the same rotation.

For a unit quaternion $q=(w,\mathbf v)$, its inverse is its conjugate:


$$
q^{-1}=q^*=(w,-\mathbf v)
$$


Quaternion multiplication and matrix conversion must be consistent with the rotation composition convention defined in Section 2.

## 4. Rigid Transformations — $\mathrm{SE}(3)$

### 4.1 Coordinate Frame Convention

A rigid transformation $T_{A\leftarrow B}$ maps coordinates expressed in frame $B$ to coordinates expressed in frame $A$:

$$
\tilde p_A = T_{A\leftarrow B}\tilde p_B
$$

The transformation is defined by a rotation $R_{A\leftarrow B}$ and a translation $t_{A\leftarrow B}$, both expressed according to the documented frame convention.

For compatible transformations, composition follows right-to-left matrix multiplication:

$$T_{A\leftarrow C}=T_{A\leftarrow B}T_{B\leftarrow C}$$

The intermediate frame identifiers must match.

## 5. Inverse Transformation

For a rotation matrix $R\in\mathrm{SO}(3)$:

$$
R^{-1}=R^\mathsf{T}
$$

For a rigid transformation:

$$
T = \begin{bmatrix}
R & t \
0 & 1
\end{bmatrix}
$$

its inverse is:

$$
T^{-1} = \begin{bmatrix}
R^\mathsf{T} & -R^\mathsf{T}t \
0 & 1
\end{bmatrix}
$$

Implementations must validate the relevant input contract before applying these identities. The transpose shortcut is valid for rotation matrices in $\mathrm{SO}(3)$, not for arbitrary $3\times3$ matrices.


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

- The default floating-point representation for the initial C++ implementation is double.
- The Python reference implementation uses float64 for numerical geometry operations unless a function explicitly specifies otherwise.
- Numerical comparisons must use explicit tolerances appropriate to the operation.
- Absolute and relative tolerances must be distinguished. When a validation function uses only absolute tolerance, its relative tolerance must be zero.
- The initial rotation-matrix validation tolerance is atol = 1e-8, configurable through the API. This is a default, not a universal error bound.
- Non-finite values must be handled explicitly.
- Invalid dimensions and degenerate geometric configurations must be rejected or reported according to the API contract.
- Validation must distinguish exact mathematical properties from approximate floating-point checks.

## 8. Testing Requirements

Core operations must be tested against mathematical properties, not solely against example outputs.

Required properties include:

- Rotation orthogonality and determinant.
- Preservation of vector norms and pairwise distances under rotations.
- Identity rotation and composition with the identity.
- Composition order and associativity, within numerical tolerance.
- Rotation-inverse consistency: $RR^\mathsf{T}=R^\mathsf{T}R=I_3$.
- Preservation of distances under rigid transformations.
- Transformation-inverse consistency.
- Quaternion normalization and equivalence of $q$ and $-q$.
- Camera projection and reprojection consistency.
- Recovery of known rigid transformations from synthetic correspondences.


## 9. Scope and Guarantees

These conventions define the intended mathematical behavior of Astra Geometry. They do not, by themselves, establish implementation correctness, numerical conditioning, execution-time bounds, or hard real-time suitability. Such properties require separate tests and analysis.
