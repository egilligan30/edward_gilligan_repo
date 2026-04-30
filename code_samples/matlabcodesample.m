%% PURPOSE
% This script visualizes and compares two dynamical systems that can be derived
% from a single scalar function H(x,y) — the "Hamiltonian".
% 
% Given H, two natural vector fields arise:
%
%   (1) Hamiltonian system:  (dx/dt, dy/dt) = ( ∂H/∂y, -∂H/∂x )
%       Trajectories orbit along level curves of H. Because H is constant along
%       solutions, this system cannot converge to a fixed
%       point or diverge to infinity.
%
%   (2) Gradient system:     (dx/dt, dy/dt) = ( ∂H/∂x,  ∂H/∂y )
%       Trajectories climb in the direction of steepest ascent of H, crossing
%       level curves perpendicularly. The sign can be flipped for gradient descent.
%
% Producing both portraits side-by-side makes the geometric duality immediately
% visible: the two families of curves are everywhere orthogonal to each other
% (except at critical points of H).

%% SCRIPT OUTLINE
%   Section 1 — User parameters: define H, domain, resolution, and plot options.
%               Edit here to experiment with different Hamiltonians or domains.
%   Section 2 — Numerical derivatives: build anonymous functions for ∂H/∂x and
%               ∂H/∂y using central finite differences.
%   Section 3 — Grid & vector fields: construct the meshgrid, evaluate H and
%               both vector fields at every grid point.
%   Section 4 — Visualization:
%       (4a) Side-by-side phase portraits — Hamiltonian on the left (contours +
%            quiver), gradient system on the right (streamlines + quiver).
%       (4b) Overlay — both families drawn on one axes to make orthogonality
%            immediately visible.

clear; clc; close all;

%% ===================== 1. USER INPUT / PARAMETERS =====================
% -----------------------------------------------------------------------
% SCALAR FUNCTION H(x,y)
%   This is the only line you need to change to study a different system.
%   Classic examples to try:
%       H = @(x,y) x.^2 + y.^2;           
%       H = @(x,y) x.^2 - y.^2;           
%       H = @(x,y) sin(x).*cos(y);        
% -----------------------------------------------------------------------
H = @(x,y) x.^2 - 2*x.*y - y.^2 + 2*x - 2*y + 2;

% -----------------------------------------------------------------------
% FINITE DIFFERENCE STEP SIZE
%   Used to numerically approximate ∂H/∂x and ∂H/∂y (see Section 2).
%   Should be small enough for accuracy but not so small that floating-point
%   cancellation dominates.
% -----------------------------------------------------------------------
eps_fd = 1e-5;

% -----------------------------------------------------------------------
% DOMAIN AND GRID RESOLUTION
%   x_range / y_range define the window shown in each plot.
%   grid_N controls how many sample points are used along each axis.
%   Larger grid_N gives smoother contours but increases memory usage.
% -----------------------------------------------------------------------
x_range = [-3, 3];
y_range = [-3, 3];
grid_N  = 400;

% -----------------------------------------------------------------------
% QUIVER DISPLAY SETTINGS
%   quiver_skip: only every k-th grid point gets an arrow; prevents the plot
%                from being buried under overlapping arrows.
%   quiver_scale: scales arrow length; smaller values shorten arrows.
%   Arrows are normalized to unit length before scaling so that the quiver
%   shows direction only, not magnitude.
% -----------------------------------------------------------------------
quiver_skip  = 20;
quiver_scale = 0.4;

% -----------------------------------------------------------------------
% CONTOUR LEVELS
%   These are the values of H at which level curves are drawn. They become
%   the Hamiltonian trajectories in the left panel and the reference curves
%   in the overlay; adjust the range to cover the domain of interest.
% -----------------------------------------------------------------------
H_levels = linspace(0.1, 16, 10);

% -----------------------------------------------------------------------
% STREAMLINE SEED POINTS (for gradient flow)
%   Streamlines are integrated forward along the gradient vector field
%   starting from these (x,y) positions.
%   Add or remove seeds to trace different regions.
% -----------------------------------------------------------------------
startX = [-2 -1.5 -1 -0.5  0  0.5  1  1.5  2 ...
          -2 -1.5 -1 -0.5  0  0.5  1  1.5  2];

startY = [-2 -2   -2 -2   -2 -2   -2 -2   -2 ...
           2  2    2  2    2  2    2  2    2];

% -----------------------------------------------------------------------
% PLOT TOGGLES
%   Set either flag to false to skip that figure.
% -----------------------------------------------------------------------
make_side_by_side = true;
make_overlay      = true;

%% ===================== 2. NUMERICAL DERIVATIVES =====================
% -----------------------------------------------------------------------
% Central finite difference formulas:
%   ∂H/∂x ≈ [ H(x + ε, y) - H(x - ε, y) ] / (2ε)
%   ∂H/∂y ≈ [ H(x, y + ε) - H(x, y - ε) ] / (2ε)
%
% Central differences are second-order accurate (error ~ ε²).
% Both Hx and Hy are returned as anonymous functions so they can be called
% on scalars, vectors, or full meshgrids with identical syntax.
% -----------------------------------------------------------------------
Hx = @(x,y) (H(x+eps_fd, y) - H(x-eps_fd, y)) / (2*eps_fd);
Hy = @(x,y) (H(x, y+eps_fd) - H(x, y-eps_fd)) / (2*eps_fd);

%% ===================== 3. GRID + VECTOR FIELDS =====================
% -----------------------------------------------------------------------
% Build a uniform rectangular grid over the specified domain.
%   xv, yv are 1-D coordinate vectors (length grid_N each).
%   meshgrid returns 2-D arrays X and Y of size (grid_N × grid_N) so that
%   every (X(i,j), Y(i,j)) pair is a distinct point in the domain.
% -----------------------------------------------------------------------
xv = linspace(x_range(1), x_range(2), grid_N);
yv = linspace(y_range(1), y_range(2), grid_N);
[X, Y] = meshgrid(xv, yv);

% Evaluate the scalar field H at every grid point.
% H_vals is used for drawing the level curves.
H_vals = H(X, Y);

% -----------------------------------------------------------------------
% Hamiltonian vector field: rotate the gradient 90° counter-clockwise.
%   (Ux, Vx) = ( ∂H/∂y, -∂H/∂x )
%   This 90° rotation is what makes trajectories flow along the level 
%   curves of H.
% -----------------------------------------------------------------------
U_ham  =  Hy(X, Y);   %  x-component of ∂H/∂y
V_ham  = -Hx(X, Y);   %  y-component of -∂H/∂x

% -----------------------------------------------------------------------
% Gradient vector field:
%   (Ux, Vx) = ( ∂H/∂x, ∂H/∂y )
% -----------------------------------------------------------------------
U_grad =  Hx(X, Y);   %  x-component of ∂H/∂x
V_grad =  Hy(X, Y);   %  y-component of ∂H/∂y

%% ===================== 4. VISUALIZATION PREP =====================
% -----------------------------------------------------------------------
% Quiver plots:
%   Using every grid point would pack arrows so densely they become
%   unreadable. Keeping only every (quiver_skip)-th row and column
%   gives a legible figure.
% -----------------------------------------------------------------------
sk = quiver_skip;

Xq   = X(1:sk:end, 1:sk:end);
Yq   = Y(1:sk:end, 1:sk:end);

Uq_h = U_ham(1:sk:end,  1:sk:end);
Vq_h = V_ham(1:sk:end,  1:sk:end);

Uq_g = U_grad(1:sk:end, 1:sk:end);
Vq_g = V_grad(1:sk:end, 1:sk:end);

% -----------------------------------------------------------------------
% Normalize arrows to unit length before passing to quiver.
%   Without normalization, arrows in high-gradient regions would dwarf those
%   in low-gradient regions and obscure the structure. The small
%   constant (1e-10) prevents division by zero near critical points.
% -----------------------------------------------------------------------
mag_h = sqrt(Uq_h.^2 + Vq_h.^2) + 1e-10;
mag_g = sqrt(Uq_g.^2 + Vq_g.^2) + 1e-10;

%% ===================== 4a. SIDE-BY-SIDE PLOTS =====================
if make_side_by_side
    figure('Color','white','Position',[100 100 1200 520]);

    % Hamiltonian system:
    subplot(1,2,1); hold on; axis equal; box on;

    % Draw level curves of H 
    contour(X, Y, H_vals, H_levels, 'LineWidth', 1.5, ...
            'LineColor', [0.18 0.53 0.82]);

    % Overlay direction arrows to show flow direction on each curve
    quiver(Xq, Yq, Uq_h./mag_h, Vq_h./mag_h, quiver_scale, ...
           'Color', [0.18 0.53 0.82 0.55]);

    % Mark the origin as a reference point
    plot(0, 0, 'ko', 'MarkerFaceColor', 'k');

    xlabel('x'); ylabel('y');
    xlim(x_range); ylim(y_range);
    title('Hamiltonian system');

    % Gradient system:
    subplot(1,2,2); hold on; axis equal; box on;

    % Draw the same level curves as dashed gray lines for reference
    contour(X, Y, H_vals, H_levels, 'LineStyle','--', ...
            'LineColor', [0.6 0.6 0.6]);

    % Normalize the gradient field before integration
    mag_g_full = sqrt(U_grad.^2 + V_grad.^2) + 1e-10;

    streamline(X, Y, ...
        U_grad./mag_g_full, V_grad./mag_g_full, ...
        startX, startY);

    % Add direction arrows
    quiver(Xq, Yq, Uq_g./mag_g, Vq_g./mag_g, quiver_scale, ...
           'Color', [0.85 0.29 0.29 0.5]);

    % Mark the origin for spatial reference
    plot(0, 0, 'ko', 'MarkerFaceColor', 'k');

    xlabel('x'); ylabel('y');
    xlim(x_range); ylim(y_range);
    title('Gradient system');

    % Shared title for both panels
    sgtitle('Orthogonal phase portrait families');
end

%% ===================== 4b. OVERLAY =====================
% Draw both on one set of axes.
% Blue = level curves of H.
% Red = gradient paths.
if make_overlay
    figure('Color','white');
    hold on; axis equal; box on;

    % Hamiltonian level curves as solid blue lines.
    % [~, hc] gets the contour object for the legend
    [~, hc] = contour(X, Y, H_vals, H_levels, 'LineWidth', 1.8);
    hc.LineColor = [0.18 0.53 0.82];

    % Gradient trajectories in red.
    h_stream2 = streamline(X, Y, U_grad, V_grad, startX, startY);
    set(h_stream2, 'Color', [0.85 0.29 0.29]);

    % Mark the origin
    plot(0, 0, 'ko', 'MarkerFaceColor', 'k');

    xlabel('x'); ylabel('y');
    xlim(x_range); ylim(y_range);

    title('Overlaid: Hamiltonian ⟂ Gradient flow');

    % Legend uses the contour object and the first streamline handle
    legend([hc, h_stream2(1)], ...
        {'Hamiltonian trajectories', 'Gradient trajectories'}, ...
        'Location', 'southeast');
end