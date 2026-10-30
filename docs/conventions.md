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


For a general (non-unit) quaternion the inverse is $q^{-1}=q^*/\lVert q\rVert^2$. The conjugate alone is an inverse only in the unit case.

Quaternion multiplication and matrix conversion must be consistent with the rotation composition convention defined in Section 2. Given $q_1$ and $q_2$, the product $q_2q_1$ represents applying $q_1$ first and $q_2$ second.

### 3.1 API conventions

- **Composition order.** The multiplication entry point mirrors `compose_rotations`: the first argument is the rotation applied first, so `quaternion_multiply(q1, q2)` returns $q_2q_1$, whose rotation matrix is $R_2R_1$. The product is not commutative and argument order is part of the contract.
- **Unit-norm requirement.** Multiplication and conjugation are algebraic operations: they require shape $(4,)$, real and finite components, but not unit norm. The product of two unit quaternions is unit, and $\lVert q_2q_1\rVert=\lVert q_2\rVert\lVert q_1\rVert$ in general. The conversion functions, which represent rotations, do require unit norm validated with a configurable absolute tolerance.
- **Conversion output representative.** Converting a rotation matrix to a quaternion returns the representative with $w\ge 0$ whenever the rotation allows it. For an exact half turn ($w=0$) both $q$ and $-q$ are equally valid and the rule cannot choose: the returned sign is an implementation detail and must not be relied upon. Sign stability in the neighbourhood of $w=0$ is not guaranteed. The rotation matrix produced by $q\to R$ does not depend on the sign of $q$.

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
R & t \\
0 & 1
\end{bmatrix}
$$

its inverse is:

$$
T^{-1} = \begin{bmatrix}
R^\mathsf{T} & -R^\mathsf{T}t \\
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

Here, $(X_c,Y_c,Z_c)$ are coordinates in the camera frame, $(u,v)$ are image coordinates, and $\lambda$ is the projective scale. For the standard pinhole camera model, $\lambda=Z_c$, assuming the stated intrinsic matrix and $Z_c\ne0$.

### 6.1 Image and pixel convention

- Image coordinates $(u,v)$ are expressed in pixels.
- The origin $(0,0)$ is the top-left corner of the image; $u$ increases to the right and $v$ increases downward (the standard image convention).
- The principal point $(c_x,c_y)$ is given in pixels in the same convention.
- The focal lengths are positive: $f_x>0,\ f_y>0$.

The last row of $K$ is $[0,0,1]$ and the entry below the diagonal is $0$: both are structure of the representation, required exactly and carrying no tolerance, in the same sense as the last row of a rigid transformation (section 7.3). The skew $s$ may be non-zero and is validated only for finiteness.

## 7. Numerical Precision and Validation

### 7.1 Floating-point representation

- The default floating-point representation for the initial C++ implementation is double.
- The Python reference implementation uses float64 for numerical geometry operations unless a function explicitly specifies otherwise.
- float64 is the default format, and it must not be taken to mean that all entries are converted to it, nor that all operations are exact.

### 7.2 Tolerance policy

- Numerical comparisons must use explicit tolerances appropriate to the operation.
- Absolute and relative tolerances must be distinguished. When a validation function uses only absolute tolerance, its relative tolerance must be zero ($\mathrm{rtol}=0$).
- The reference absolute tolerance is `atol = 1e-8`, configurable through the API and shared by the SO(3), quaternion, and SE(3) validations. It is a default, not a universal error bound.
- **Tolerance parameters are keyword-only.** A tolerance is always passed by keyword, never positionally. A positional float after the primary argument is a silent misbinding — it can be read as a second data argument — and it is invisible to round-trip tests. Tolerance parameters are therefore declared keyword-only (after a bare `*`), so that every call site is explicit and greppable.
- A non-finite or negative tolerance is a violated contract: predicates return `False`; constructors and operations raise `ValueError`.

### 7.3 Homogeneous representation and structural exactness

- Rigid transformations are represented as $4\times4$ homogeneous matrices $T=\begin{bmatrix}R&t\\0&1\end{bmatrix}$, with $R\in\mathrm{SO}(3)$ and $t$ expressed in the destination frame.
- The last row $[0,0,0,1]$ is structure of the representation, not an approximated geometric property. It is required exactly, with no tolerance: a matrix with noise in that row must be repaired explicitly, not validated. `atol` governs only the rotation block (orthogonality and determinant), reusing the SO(3) validation.
- Requiring the last row exactly is what makes the matrix product preserve it exactly for two valid transformations: exact structure is closed under exact operations.
- General rule: a geometric tolerance must never be turned into permission to admit incorrect structure. Structure is validated exactly; geometry is validated approximately.

### 7.4 Input handling

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
- Quaternion composition consistent with matrix composition, including argument order.
- Camera projection and reprojection consistency.
- Recovery of known rigid transformations from synthetic correspondences.


## 9. Scope and Guarantees

These conventions define the intended mathematical behavior of Astra Geometry. They do not, by themselves, establish implementation correctness, numerical conditioning, execution-time bounds, or hard real-time suitability. Such properties require separate tests and analysis.
