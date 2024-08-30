library(ggplot2)
library(dplyr)

# Create a DataFrame from your data
data <- data.frame(
  ModelPerformance = rep(c("Model score", "Validation score"), each = 6),
  Metric = rep(c("Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Specificity"), times = 2),
  Score = c(0.9752, 0.9884, 0.9607, 0.9743, 0.975, 0.9892, 0.95, 0.95, 0.95, 0.95, 0.99, 0.91)
)

# Convert ModelPerformance to a factor for proper ordering in the plot
data$ModelPerformance <- factor(data$ModelPerformance, levels = c("Model score", "Validation score"))

# Font size variables
title_font_size <- 22
axis_text_font_size <- 20
axis_title_font_size <- 22
legend_text_font_size <- 20 # Add variable for legend text font size

# Plotting
p <- ggplot(data, aes(x = Metric, y = Score, fill = ModelPerformance)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.5), width = 0.4) +
  labs(x = "Metric", y = "Score", title = "Model and Validation Scores") +
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
