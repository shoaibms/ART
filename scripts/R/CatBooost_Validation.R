library(ggplot2)
library(dplyr)

# Create a DataFrame from your data
data <- data.frame(
  Algorithms = rep("CatBoost", 12),
  Dataset = rep(c("Combine", "Validate"), each = 6),
  Metric = rep(c("Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Specificity"), times = 2),
  Score = c(0.96, 0.96, 0.96, 0.96, 0.99, 0.96, 0.5, 0.25, 0.5, 0.3333, 0.5, 0)
)

# Convert Dataset to a factor for proper ordering in the plot
data$Dataset <- factor(data$Dataset, levels = c("Combine", "Validate"))

# Font size variables
title_font_size <- 24
axis_text_font_size <- 24
axis_title_font_size <- 24
legend_text_font_size <- 24 # Add variable for legend text font size

# Plotting
p <- ggplot(data, aes(x = Metric, y = Score, fill = Dataset)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.5), width = 0.4) +
  labs(x = "Metric", y = "Score", title = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = axis_text_font_size),
        axis.text.y = element_text(size = axis_text_font_size),
        axis.title.x = element_text(size = axis_title_font_size),
        axis.title.y = element_text(size = axis_title_font_size),
        plot.title = element_text(hjust = 0.5, size = title_font_size),
        legend.position = "bottom",  # Position the legend at the bottom
        legend.title = element_blank(),
        legend.text = element_text(size = legend_text_font_size),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black")) +
  scale_fill_manual(values = c("#f7d0d0", "#fadce8"))

# Display the plot
print(p)
