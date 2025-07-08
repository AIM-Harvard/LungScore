"""
Deep-learning biomarker for Lung Health stats_analysis - Boxplot

# The code and data of this repository are intended to promote transparent and reproducible research
# of the paper -- Deep Learning-based Lung Health Quantification on Computed Tomography in Adults

# THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


"""
Example usage at the end of this file
"""


# load libraries
library(meta)
require(survival)
library(survminer) 
library(dplyr)
library(grid)
library(gridExtra)
library(meta)
library(naniar)
library(ggplot2)
library(scales)
library(ggpubr)
library(ggplot2)
library(dplyr)
library(ggbeeswarm) 
library(ggsignif)   

############################
# Boxplot function
#############################
plot_boxplot <- function(data, categorical_var, continuous_var, output_file, labels_colors, x_label = "PACK-YEARS", y_label = "Lung health", width = 5, height = 5, ref.group="0", y_lim = c(0, 1)) {
  # Helper function for adding n = labels
  give.n.eq <- function(yx){
    inner <- function(x){
      return(list(y = yx, label = paste0("n = ", length(x))))
    }
    return(inner)
  }
  
  # Apply themes
  theme_set(theme_classic())
  theme_update(axis.text = element_text(color = "black"))
  
  # Create the plot
  box_plot <- data %>%
    filter(!is.na(!!sym(categorical_var))) %>%
    ggplot(aes(x = !!sym(categorical_var), y = !!sym(continuous_var), fill = !!sym(categorical_var))) +
    geom_boxplot(outlier.shape = NA) +
    #geom_quasirandom(width = 0.2, size = 0.5) +
    geom_quasirandom(width = 0.2, size = 0.1) +
    stat_compare_means(method = "wilcox", label = 'p.signif', label.x.npc = 0.5, hjust = 0.5, label.y = 1, step.increase = 0.1, ref.group = ref.group) +
    stat_summary(fun.data = give.n.eq(0), geom = "text", color ="black") + # Adds n = at position 0
    #coord_cartesian(ylim = c(0, 105)) +
    coord_cartesian(ylim = y_lim) +
    labs(y = y_label, x = x_label) +
    #scale_fill_manual(values = labels_colors) +
    scale_fill_brewer(palette = "Set3") +
    # scale_fill random brewer palette
    
    theme(legend.position = "none")
  
  # Save and return the plot
  ggsave(output_file, box_plot, width = width, height = height)
  return(box_plot)
}



####################################################

# Example usage for oevrall survival in NLST
# read data
df_NLST <- read.csv('/mnt/data/NLST_test.csv')

# boxplot NLST pack-years - Figure 5d
# packyears groups
quantiles <- quantile(df_NLST$pkyr, probs = c(0.25, 0.5, 0.75), na.rm = TRUE)

df_NLST$pack_years_groups <- cut(
  df_NLST$pkyr,
  breaks = c(-Inf, quantiles[1], quantiles[2], quantiles[3], Inf),
  labels = c("<39", "39-48", "48-66", ">66"),
  include.lowest = TRUE,
  right = TRUE
)

df_NLST$pack_years_groups <- factor(df_NLST$pack_years_groups)

# labels_colors
labels_colors <- c( 
  "<39" = "#8FBCBB", "39-48" = "#88C0D0" , "48-66" = "#81A1C1", ">66" = "#5E81AC"
)

# pack years box plot
result <- plot_boxplot(df_NLST, 
                               "pack_years_groups", 
                               "lunghealth_cont", 
                               "mnt/outcome_analysis/Figures_LH/Boxplot_packyears_NLST.pdf", 
                               labels_colors = labels_colors,
                               x_label = "Pack-years groups\n (NLST)",
                               y_label = "Lung health score", 
                               height = 3, 
                               width = 3.2,
                               ref.group="<39") 
