library(ggplot2)
library(ggbump)
library(dplyr)
library(tidyr)

# Recreate the data
data <- data.frame(
  Genotype = c("DT_3", "DT_2", "DT_1", "DS_1", "DS_3", "DS_2"),
  Rank_1 = c(1, 2, 3, 4, 5, 6),
  Rank_2 = c(1, 2, 3, 4, 6, 5),
  Rank_3 = c(1, 2, 3, 4, 5, 6)
)

# Transform data to long format and set the order of Rank_Type
data_long <- data %>%
  pivot_longer(cols = -Genotype, names_to = "Rank_Type", values_to = "Rank") %>%
  mutate(Rank_Type = factor(Rank_Type, levels = c("Rank_1", "Rank_2", "Rank_3")))# RD_rank= Rank_1, SSI_rank = Rank_2, COMB_rank = Rank_3

custom_colors <- c("DT_1" = "#78ccb5", # Gladius = DT_1
                   "DT_2" = "#66dec9", # DAS5_005489 = DT_2
                   "DT_3" = "#10b397", # DAS5_CALINGIRI = DT_3
                   "DS_1" = "#f7ba36", # Forrest = DS_1
                   "DS_2" = "#f5c47a",# Hartog = DS_2
                   "DS_3" = "#ada358")# DAS5_003811 = DS_3

# Create the bump chart with custom colors
ggplot(data_long, aes(x = Rank_Type, y = Rank, group = Genotype, color = Genotype)) +
  geom_bump(size = 2) +  # Adjust line size for emphasis
  geom_point(size = 6) +   # Emphasize points
  geom_text(aes(label = ifelse(Rank_Type != "Rank_2", Genotype, "")), nudge_y = 0.4, size = 7) +  # Increase label font size
  scale_y_reverse() +   # Reverse y-axis for typical bump chart orientation
  scale_color_manual(values = custom_colors) + # Apply custom colors
  labs(title = "",
       x = "Rank Type",
       y = "Rank") +
  theme_minimal() +  # Simple, clean theme
  theme(
    legend.position = "none",  # Remove legend for cleaner look
    axis.title.x = element_text(size = 22),  # Increase x-axis title font size
    axis.title.y = element_text(size = 22),  # Increase y-axis title font size
    axis.text.x = element_text(angle = 45, hjust = 1, size = 20),  # Increase x-axis text
    axis.text.y = element_text(size = 20)
  )

