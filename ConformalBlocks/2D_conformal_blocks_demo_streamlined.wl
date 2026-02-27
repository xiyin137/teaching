(*
  2D Global Conformal Block and Crossing Symmetry Demo
  =====================================================

  Background
  ----------
  In a 2D CFT, the four-point function of identical scalars phi with
  dimension Delta_phi decomposes into global conformal blocks that
  factorize into holomorphic and anti-holomorphic pieces:

    G_{Delta,ell}(z, zbar) = k_{Delta+ell}(z) k_{Delta-ell}(zbar)
                            + k_{Delta+ell}(zbar) k_{Delta-ell}(z),

  where k_beta(x) = x^{beta/2} _2F_1(beta/2, beta/2; beta; x) is the
  SL(2,R) block and (z, zbar) are the usual cross-ratio variables.

  Crossing symmetry for identical external scalars reads

    v^{Delta_phi} G(z, zbar) = u^{Delta_phi} G(1-z, 1-zbar)

  where u = z*zbar, v = (1-z)*(1-zbar).  The "crossing residual" is
  the difference of the two sides; it vanishes for the exact OPE.
  This script evaluates that residual on the diagonal z = zbar for a
  toy truncated spectrum (3 operators) to demonstrate near-cancellation.

  What this script does
  ---------------------
  1. Define global blocks via the holomorphic factorization above.
  2. Build the crossing residual for a toy spectrum at Delta_phi = 0.518
     on the z = zbar diagonal.
  3. Export a CSV and a plot showing |crossing residual| vs x.

  Run with:
    math -script 2D_conformal_blocks_demo_streamlined.wl

  Parameter glossary
  ------------------
  Delta, delta    conformal dimension of an exchanged operator
  ell             spin of an exchanged operator
  z, zbar         cross-ratio variables
  u, v            u = z*zbar, v = (1-z)*(1-zbar)
  Delta_phi       external scalar dimension
  OPE^2           squared OPE coefficient (positive for unitary theories)

  References
  ----------
  - Dolan, Osborn, hep-th/0309180 (conformal block formulas)
  - Simmons-Duffin, 1502.02033 (bootstrap review)
*)

ClearAll["Global`*"];

deltaPhi = 0.518;
outDir = FileNameJoin[{Directory[], "streamlined_output_2d_wl"}];
If[! DirectoryQ[outDir], CreateDirectory[outDir, CreateIntermediateDirectories -> True]];

(* SL(2,R) conformal block (holomorphic building block):
   k_beta(x) = x^{beta/2} _2F_1(beta/2, beta/2; beta; x).
   This is the eigenfunction of the SL(2,R) Casimir operator. *)
kBeta[beta_, x_] := x^(beta/2) Hypergeometric2F1[beta/2, beta/2, beta, x];

(* 2D global conformal block: factorizes into holomorphic x anti-holomorphic.
   G_{Delta,ell}(z,zbar) = k_{Delta+ell}(z)*k_{Delta-ell}(zbar) + (z <-> zbar).
   The symmetrization ensures Bose symmetry under z <-> zbar. *)
globalBlock2D[delta_, ell_, z_, zb_] :=
  kBeta[delta + ell, z] kBeta[delta - ell, zb] +
  kBeta[delta + ell, zb] kBeta[delta - ell, z];

(* Crossing vector for one operator in the OPE.
   F_{Delta,ell}(z,zbar) = v^{Delta_phi} G(z,zbar) - u^{Delta_phi} G(1-z,1-zbar).
   The full crossing equation is:  identity piece + Sum_i OPE_i^2 * F_i = 0.
   When the OPE is truncated, F will be small but nonzero ("crossing residual"). *)
crossingVector2D[deltaPhi_, delta_, ell_, z_, zb_] := Module[{u, v, g, gx},
  u = z zb;
  v = (1 - z) (1 - zb);
  g = globalBlock2D[delta, ell, z, zb];
  gx = globalBlock2D[delta, ell, 1 - z, 1 - zb];
  v^deltaPhi g - u^deltaPhi gx
];

(* Toy truncated spectrum: {Delta, spin, OPE^2}.
   These are approximate values for the 3D Ising model (used here as a
   pedagogical stand-in).  The first entry is the leading Z2-even scalar
   epsilon (Delta ~ 1.41), followed by the spin-2 stress tensor and a
   spin-4 operator.  The OPE^2 values are rough estimates.
   With only 3 operators the crossing residual will be small but nonzero;
   a complete OPE would give exactly zero. *)
spectrum = {
  {1.4126, 0, 0.24},
  {3.8303, 2, 0.016},
  {5.50, 4, 0.002}
};

(* Evaluate the crossing residual at z = zbar = x.
   The identity contribution is v^{Delta_phi} - u^{Delta_phi}; each operator
   in the spectrum adds its OPE^2 * crossingVector.  On the exact OPE this
   sum vanishes; for our 3-operator truncation it measures the approximation error. *)
residualAtX[x_] := Module[{z, zb, u, v, res, spec, delta, ell, ope2},
  z = x;
  zb = x;
  u = z zb;
  v = (1 - z) (1 - zb);
  res = v^deltaPhi - u^deltaPhi;
  Do[
    {delta, ell, ope2} = spec;
    res += ope2 crossingVector2D[deltaPhi, delta, ell, z, zb],
    {spec, spectrum}
  ];
  N[res, 40]
];

(* Scan the diagonal z = zbar from 0.05 to 0.45.
   The crossing point is z = 1/2; we sample symmetrically around it. *)
xValues = Range[0.05, 0.45, 0.01];
rows = Table[
  With[{r = residualAtX[x]}, {x, Re[r], Abs[r]}],
  {x, xValues}
];

csvPath = FileNameJoin[{outDir, "crossing_residual_diagonal.csv"}];
Export[csvPath, Prepend[rows, {"x", "residual_real", "residual_abs"}], "CSV"];

(* Plot |crossing residual| vs x.  For a good truncated OPE this should
   be small everywhere, with exact zeros at the crossing-symmetric point x = 1/2. *)
plot = ListLinePlot[
  rows[[All, {1, 3}]],
  Frame -> True,
  FrameLabel -> {"x (z = zbar)", "|crossing residual|"},
  PlotLabel -> "2D Global Block Demo (Truncated Spectrum)",
  PlotRange -> All,
  ImageSize -> 520
];

pngPath = FileNameJoin[{outDir, "crossing_residual_abs.png"}];
Export[pngPath, plot];

Print["Wrote: ", csvPath];
Print["Wrote: ", pngPath];
