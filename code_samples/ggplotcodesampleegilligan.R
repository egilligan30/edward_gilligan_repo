# ------------------------------------------------------------------
# Purpose:
# Visualize how 2026 NCAA tournament teams are expected to perform
# relative to historical benchmarks by seed. Monte Carlo simulated
# win projections are compared against historical average performance
# and interquartile ranges to assess whether teams are over- or
# under-performing relative to typical seed expectations.
#
# Outline:
# - Compute Monte Carlo projected wins for each team
# - Merge projections with team metadata (seeds, logos)
# - Build historical benchmark of wins by seed (mean + IQR)
# - Combine projections with historical seed performance
# - Create jittered positions for clearer visualization
# - Plot: historical distribution (ribbon + line) + team projections (logos)
# - Save final figure
# ------------------------------------------------------------------

# Load library for embedding images (team logos) in ggplot
library(ggimage)

# Average simulated wins across all Monte Carlo runs for each team
mc_proj <- data.frame(
  team         = rownames(wins_mat),   # Team names from simulation matrix
  mc_proj_wins = rowMeans(wins_mat)    # Mean projected wins
)

# Merge projections with team-level data (e.g., seed, logo)
combined_seed <- final_scores %>%
  left_join(mc_proj, by = "team")

# Construct historical distribution of tournament wins by seed
hist_seed_perf <- df_hist %>%
  mutate(
    # Total wins per tournament (missing rounds treated as 0)
    wins_total =
      replace_na(W_64, 0) +
      replace_na(W_32, 0) +
      replace_na(W_16, 0) +
      replace_na(W_8,  0) +
      replace_na(W_4,  0) +
      replace_na(W_2,  0)
  ) %>%
  group_by(seed) %>%
  summarise(
    mean_wins = mean(wins_total),         # Average wins by seed
    p25       = quantile(wins_total, 0.25), # 25th percentile
    p75       = quantile(wins_total, 0.75), # 75th percentile
    .groups   = "drop"
  )

# Combine projections with historical seed benchmarks
seed_plot <- combined_seed %>%
  filter(!is.na(seed)) %>%                      # Keep only seeded teams
  left_join(hist_seed_perf, by = "seed") %>%
  mutate(
    # Add small jitter to avoid overplotting
    seed_jitter = seed + runif(n(), -0.15, 0.15),
    wins_jitter = pmax(0, mc_proj_wins + runif(n(), -0.04, 0.04)) # Ensure non-negative
  )

# Build visualization
p_seed_projection <- ggplot() +
  
  # Historical interquartile range (shaded band)
  geom_ribbon(data = hist_seed_perf,
              aes(x = seed, ymin = p25, ymax = p75),
              fill = "grey80", alpha = 0.45) +
  
  # Historical mean wins by seed
  geom_line(data = hist_seed_perf,
            aes(x = seed, y = mean_wins),
            color = "black", linewidth = 1.2) +
  
  # Team projections plotted as logos
  geom_image(data = seed_plot,
             aes(x = seed_jitter, y = wins_jitter, image = logo),
             size = 0.055) +
  
  # Axis scaling
  scale_x_continuous(breaks = 1:16, limits = c(0.5, 16.5)) +
  scale_y_continuous(breaks = seq(0, 6, 1), limits = c(0, 6)) +
  
  # Labels and annotations
  labs(
    title    = "Seed vs Projected Tournament Wins — 2026 NCAA Tournament",
    subtitle = "Logos show Monte Carlo projected wins; black line shows historical seed expectation",
    x        = "Seed",
    y        = "Projected Tournament Wins",
    caption  = paste0(
      "Historical baseline: 2011–2025 tournaments\n",
      "Grey band = historical interquartile range of wins for that seed"
    )
  ) +
  
  # Styling
  theme_minimal(base_size = 20) +
  theme(
    plot.title    = element_text(face = "bold", size = 32, color = "black"),
    plot.subtitle = element_text(size = 22, color = "grey30"),
    plot.caption  = element_text(size = 18, color = "grey40"),
    axis.title.x  = element_text(size = 22, face = "bold"),
    axis.title.y  = element_text(size = 22, face = "bold"),
    axis.text     = element_text(size = 18),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#E6E6E6", linewidth = 0.7)
  )

# Render plot
print(p_seed_projection)

# Save plot to file
ggsave("seed_vs_projected_wins_2026.png", p_seed_projection,
       width = 13, height = 7, dpi = 200, bg = "white")