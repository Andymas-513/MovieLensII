#* AUTHOR NOTES *
#* Thank you for reading my code, a couple of notes before you deep dive
#* The entire code can take more than 4 hours to run, and a laptop would easily crush
#* Therefore, if you want to run the code I recommend:
#* 1) clear your R Studio session
#* 2) review the code in small sections
#* 3) be sure you have time to run the ML algorithms.
#* ### Warning notes in these sections indicate the approx. running time ###
#* 4) R Studio will take time to run the code, make sure your laptop is active and please be patient
#* 5) Errors about memory space can occur
#* 6) Have FUN

### CODE as of Sep 2, 2023 7:49 am AM ###

# INPUT CODE FROM EDX #

##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

# Get the current script's directory
script_directory <- dirname(rstudioapi::getSourceEditorContext()$path)

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later

# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

# NOA: Construct file paths based on my script's directory
script_directory <- dirname(rstudioapi::getActiveDocumentContext()$path)
rds_file_path_edx <- file.path(script_directory, "edx_data.rds")
rds_file_path_final <- file.path(script_directory, "final_holdout_test_data.rds")

# NOA: Save data frames as RDS files
saveRDS(edx, rds_file_path_edx)
saveRDS(final_holdout_test, rds_file_path_final)

# Clear workspace to free up memory
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Load data frames from RDS files
edx <- readRDS(rds_file_path_edx)
final_holdout_test <- readRDS(rds_file_path_final)

#NOTE: Code tested as of Aug 21, 2023 1:19 pm AM

### CODE ###

#*********************************************************************
#* 1. Data Exploration
#*********************************************************************

# Load tidyverse
library(tidyverse)
library(dplyr)

# 1.1. Explore database
# a. Summary
summary(edx)

# b. Structure and display first few rows
str(edx)
head(edx)

#1.2. Plot data distribution using my favorite plot theme from The Economist

# install and load packages
if (!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
library(ggthemes)
if (!require(tidytext)) install.packages("tidytext", repos = "http://cran.us.r-project.org")
library(tidytext)

# a. Plot distribution of ratings

plot_0_0 <- ggplot(edx, aes(x = rating)) +
  geom_histogram(binwidth = 0.5, fill = "#028090", color = "white") +
  theme_economist_white() +
  labs(title = "Distribution of Ratings", x = "Rating", y = "Count")

# Save the plot as an image file
plot_file_path_0_0 <- file.path(script_directory, "distribution_of_ratings.png")
ggsave(plot_file_path_0_0, plot_0_0, width = 8, height = 6)

# b. Plot average ratings per user

# Calculate average rating per user
mean_ratings_per_user <- edx %>%
  group_by(userId) %>%
  summarize(avg_rating = mean(rating), num_ratings = n())

plot_0_1 <- ggplot(mean_ratings_per_user, aes(x = avg_rating)) +
  geom_histogram(binwidth = 0.1, fill = "#028090", color = "grey") +
  theme_economist_white() +
  labs(title = "Average Ratings per User", x = "Average Rating", y = "Count")

# Save the plot as an image file
plot_file_path_0_1 <- file.path(script_directory, "average_rating_per_user.png")
ggsave(plot_file_path_0_1, plot_0_1, width = 8, height = 6)

# c. Plot distribution of movie by genre

# Explore genres
genres <- edx %>%
  distinct(movieId, genres) %>%
  separate_rows(genres, sep = "\\|") %>%
  filter(genres != "") %>%
  count(genres, sort = TRUE)

plot_0_2 <- ggplot(genres, aes(x = reorder(genres, n), y = n)) +
  geom_bar(stat = "identity", fill = "#456990") +
  coord_flip() +
  theme_economist_white() +
  labs(title = "Distribution of Movies by Genre", x = "Genre", y = "Count")

# Save the plot as an image file
plot_file_path_0_2 <- file.path(script_directory, "Distribution of Movies by Genre.png")
ggsave(plot_file_path_0_2, plot_0_2, width = 8, height = 6)

#* NOTE: Code tested as of Aug 21, 2023, 2:15 pm AM
#* NOTE: Plots tested and saved as of Aug 26, 2023 7:13 pm AM

# Put plots in a grid
install.packages("gridExtra")
library(gridExtra)

# Arrange the plots in a 2x1 matrix layout
arranged_plots_0 <- grid.arrange(plot_0_0, plot_0_1, ncol = 2)

# Save the matrix as an image file
matrix_file_path_0 <- file.path(script_directory, "Distribution and Average of Ratings.png")
ggsave(matrix_file_path_0, arranged_plots_0, width = 8, height = 6)

#**************************************************************
#* 2. Model development
#**************************************************************

# 2.0.1 Create the RMSE function

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# 2.0.2 Make the edx file size okay for my laptop processor with random sample
#* NOTE: I need to perform this random sampling, as my laptop crashes
#* when using logistic regression, ML algorithms and sometimes simple model
#* with data sets of >1MM records

# Load libraries
if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
library(tidyverse)

# Set a seed
set.seed(1974)

# Create a random sample of maximum 200,000 records
max_records <- 200000
sampled_edx <- edx %>% sample_n(max_records)

# Compare edx and sample_edx files to evaluate similarity
summary(edx)
summary(sampled_edx)

# Save data frames as RDS files
rds_file_path_sampled_edx <- file.path(script_directory, "sampled_edx_data.rds")
saveRDS(sampled_edx, rds_file_path_sampled_edx)

# Data frames saved as of Aug 26, 2023 7:26 pm AM

# Create training and test files in the 200K sample file

set.seed(1974)
indexes <- split(1:nrow(sampled_edx), sampled_edx$userId)
test_ind <- sapply(indexes, function(ind) sample(ind, ceiling(length(ind) * 0.2))) |>
  unlist(use.names = TRUE) |> sort()
test_set <- sampled_edx[test_ind,]
train_set <- sampled_edx[-test_ind,]

test_set <- test_set |> 
  semi_join(train_set, by = "movieId")
train_set <- train_set |> 
  semi_join(test_set, by = "movieId")

# 2.1. Traditional Models

# 2.1.0. Create a benchmark model, i.e. simple average model

mu <- mean(train_set$rating , na.rm = TRUE)
mu
#AVG RATING> [1] 3.517708

naive_rmse <- RMSE(test_set$rating, mu)
naive_rmse
#RMSE: [1] 1.059203
#* NOTE: Models must have an RMSE <1.059203 to be more effective than a naive model

# Plot RMSE output

# Actual vs Predicted plot

library(ggplot2)
mu <- mean(train_set$rating, na.rm = TRUE)
naive_rmse <- 1.059203

# Create a data frame for plotting
plot_data <- data.frame(
  actual = test_set$rating,
  predicted = rep(mu, nrow(test_set))
)

# Create the scatter plot
plot <- ggplot(plot_data, aes(x = actual, y = predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(
    x = "Actual Rating",
    y = "Predicted Rating",
    title = "Actual vs. Predicted Ratings (Simple Average Model)",
    subtitle = paste("Naive RMSE:", format(round(naive_rmse, 2), nsmall = 2))
  ) +
  theme_economist_white()

# Save the plot as an image file
plot_file_path <- file.path(script_directory, "actual_vs_predicted_plot.png")
ggsave(plot_file_path, plot, width = 8, height = 6)

# Additional code to plot RMSE results tested as of Aug 25, 2023 4:05 pm AM

# Distribution of Residuals plot

# Calculate residuals
residuals <- test_set$rating - mu

# Create a histogram of residuals
plot_2 <- ggplot(data = data.frame(residuals), aes(x = residuals)) +
  geom_histogram(binwidth = 0.5, fill = "#456990", color = "grey") +
  labs(
    title = "Distribution of Residuals (Benchmark Model)",
    x = "Residuals",
    y = "Frequency"
  ) +
  theme_economist_white()+
  coord_cartesian(xlim = c(min(-3), max(3)))

plot_file_path_2 <- file.path(script_directory, "residuals_plot.png")
ggsave(plot_file_path_2, plot_2, width = 8, height = 6)

# Additional code to plot distribution of residuals tested as of Aug 25, 2023 4:19 pm AM

# 2.1.1 Calculate feature-specific effects to improve model

# Start by accounting for the same biases & effects in the case-study:

# Data pre-processing #SOURCE: http://rafalab.dfci.harvard.edu/dsbook/large-datasets.html#regularization #

y <- select(train_set, movieId, userId, rating) |>
  pivot_wider(names_from = movieId, values_from = rating) 
rnames <- y$userId
y <- as.matrix(y[,-1])
rownames(y) <- rnames

movie_map <- train_set |> select(movieId, title) |> distinct(movieId, .keep_all = TRUE)

# a. Movie Effects

b_i <- colMeans(y - mu, na.rm = TRUE)
plot_2_0 <- qplot(b_i, bins = 10,
      xlab = "Movie Effect (b_i)", ylab = "Frequency",
      main = "Histogram of Movie Effects",
      fill = I("#456990")) +
  theme_economist_white()

plot_file_path_2_0 <- file.path(script_directory, "movie_effect.png")
ggsave(plot_file_path_2_0, plot_2_0, width = 8, height = 6)

fit_movies <- data.frame(movieId = as.integer(colnames(y)), 
                         mu = mu, b_i = b_i)

left_join(test_set, fit_movies, by = "movieId") |> 
  mutate(pred = mu + b_i) |> 
  summarize(rmse = RMSE(rating, pred))

# RMSE
# rmse
# 0.9708004
#* NOTE: As demonstrated in the study-case, RMSE improves by including Movie Effect to predict
#* the rating

# Include additional biases & effects

# b. user effect only

b_u_o <- rowMeans(y, na.rm = TRUE)
qplot(b_u_o, bins = 30, color = I("black"))

b_u_o <- rowMeans(sweep(y - mu, 2, b_i), na.rm = TRUE)

fit_users <- data.frame(userId = as.integer(rownames(y)),
                        b_u_o = b_u_o)

optimal_value = mu

left_join(test_set, fit_movies, by = "movieId") |> 
  left_join(fit_users, by = "userId") |> 
  mutate(pred = mu + b_u_o) |>
  mutate(pred = ifelse(is.na(pred), optimal_value, pred)) |>
  summarize(rmse = RMSE(rating, pred))

# RMSE: 1.122547
# NOTE: RMSE worsens and the time to process increases 

# c. Include day of the week effect
# NOTE: Distribution of average scores varies based on day of the week

# Load libraries
if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(lubridate)

# Set a seed
set.seed(1974)

# Data engineering:
# Convert timestamp to datetime and extract day of the week

sampled_edx$timestamp <- as.POSIXlt(sampled_edx$timestamp, origin = "1970-01-01", tz = "UTC")
sampled_edx$day_of_week <- weekdays(sampled_edx$timestamp)

# Calculate mean rating for each day of the week
day_effect <- sampled_edx %>%
  group_by(day_of_week) %>%
  summarize(mean_rating = mean(rating, na.rm = TRUE))
day_effect

# Save the table to use in RMD
csv_file_path_1 <- file.path(script_directory, "day_effect.csv")
write.csv(file = csv_file_path_1, day_effect, row.names = FALSE)

# Calculate the overall mean rating
mu <- mean(sampled_edx$rating, na.rm = TRUE)

# Calculate the effect of the day of the week
day_effect$day_effect <- day_effect$mean_rating - mu

# Calculate movie effects and predict ratings
b_d <- colMeans(y - mu, na.rm = TRUE)

fit_movies <- data.frame(movieId = as.integer(colnames(y)), 
                         mu = mu, b_d = b_d)

# Add day_of_week to test_set
test_set$day_of_week <- 
  weekdays(as.POSIXlt(test_set$timestamp, origin = "1970-01-01", tz = "UTC"))

# Joining with day effect
predicted_ratings <- left_join(test_set, fit_movies, by = "movieId") %>%
  left_join(day_effect, by = c("day_of_week" = "day_of_week")) %>%
  mutate(pred = mu + b_d + day_effect) %>%
  summarize(rmse = RMSE(rating, pred))

print(predicted_ratings)

# RMSE improves & processing time reduces substantially
#> RMSE: 0.9707926

# d. combining movieId and day-of-the-week effects

# Load libraries
if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(lubridate)

# Convert timestamp to datetime and extract day of the week
sampled_edx$timestamp <- as.POSIXlt(sampled_edx$timestamp, origin = "1970-01-01", tz = "UTC")
sampled_edx$day_of_week <- weekdays(sampled_edx$timestamp)

# Calculate mean rating for each day of the week
day_effect <- sampled_edx %>%
  group_by(day_of_week) %>%
  summarize(mean_rating = mean(rating, na.rm = TRUE))

# Calculate the overall mean rating
mu <- mean(sampled_edx$rating, na.rm = TRUE)

# Calculate the effect of the day of the week
day_effect$day_effect <- day_effect$mean_rating - mu

# Calculate movie effects and predict ratings
b_i <- colMeans(y - mu, na.rm = TRUE)

fit_movies <- data.frame(movieId = as.integer(colnames(y)), 
                         mu = mu, b_i = b_i)

# Add day_of_week to test_set
test_set$day_of_week <- weekdays(as.POSIXlt(test_set$timestamp, origin = "1970-01-01", tz = "UTC"))

# Joining with day effect and movie effect
predicted_ratings <- left_join(test_set, fit_movies, by = "movieId") %>%
  left_join(day_effect, by = c("day_of_week" = "day_of_week")) %>%
  mutate(pred = mu + b_i + day_effect) %>%
  summarize(rmse = RMSE(rating, pred))

print(predicted_ratings)

# RMSE: 0.9707926
# NOTE: RMSE is similar to the previous model, but no improvement

# e including the user effect

# Load libraries
library(tidyverse)
library(lubridate)

# Convert timestamp to datetime and extract day of the week
sampled_edx$timestamp <- as.POSIXlt(sampled_edx$timestamp, origin = "1970-01-01", tz = "UTC")
sampled_edx$day_of_week <- weekdays(sampled_edx$timestamp)

# Calculate mean rating for each day of the week
day_effect <- sampled_edx %>%
  group_by(day_of_week) %>%
  summarize(mean_rating = mean(rating, na.rm = TRUE))

# Calculate the overall mean rating and day-of-the-week effects
mu <- mean(sampled_edx$rating, na.rm = TRUE)
b_d <- day_effect$mean_rating - mu
fit_days <- data.frame(day_of_week = unique(sampled_edx$day_of_week), b_d = b_d)

# Add day_of_week to test_set
test_set$day_of_week <- weekdays(as.POSIXlt(test_set$timestamp, origin = "1970-01-01", tz = "UTC"))

# Calculate user effects
b_u <- rowMeans(y - mu, na.rm = TRUE)
test_set$user_effect <- b_u[test_set$userId]

# Joining with day effect, movie effect, and user effect
predicted_ratings <- test_set %>%
  left_join(fit_movies, by = "movieId") %>%
  left_join(fit_days, by = "day_of_week") %>%
  mutate(pred = mu + b_i + b_d + user_effect) %>%
  mutate(pred = if_else(is.na(pred), mu + b_i + b_d, pred)) %>%
  summarize(rmse = RMSE(rating, pred))

print(predicted_ratings)

# RMSE: 1.222385
# NOTE: RMSE worsens

# NOTE: Code tested as of Aug 21, 2023 9:55 pm AM

# 2.2. Models based on ML Algorithms

# g Model trained using Xgboost
### WARNING: Running the code to train the Xgboost model takes 3 hours including tuning ###

# Load libraries
if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if (!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(lubridate)
library(xgboost)

# Set a seed
set.seed(1974)

# Convert timestamp to datetime and extract day of the week
sampled_edx$timestamp <- as.POSIXlt(sampled_edx$timestamp, origin = "1970-01-01", tz = "UTC")
sampled_edx$day_of_week <- weekdays(sampled_edx$timestamp)

# Calculate mean rating for each day of the week
day_effect <- sampled_edx %>%
  group_by(day_of_week) %>%
  summarize(mean_rating = mean(rating, na.rm = TRUE))

# Calculate the overall mean rating
mu <- mean(sampled_edx$rating, na.rm = TRUE)

# Calculate the effect of the day of the week
day_effect$day_effect <- day_effect$mean_rating - mu

# Calculate movie effects
b_i <- colMeans(y - mu, na.rm = TRUE)

fit_movies <- data.frame(movieId = as.integer(colnames(y)), 
                         mu = mu, b_i = b_i)

# Add day_of_week to test_set
test_set$day_of_week <- weekdays(as.POSIXlt(test_set$timestamp, origin = "1970-01-01", tz = "UTC"))

# Joining with day effect and movie effect
test_data <- left_join(test_set, fit_movies, by = "movieId") %>%
  left_join(day_effect, by = c("day_of_week" = "day_of_week"))

# Prepare data for XGBoost
xgb_data <- xgb.DMatrix(
  data = model.matrix(~ . - rating, data = train_set), label = train_set$rating)

# Train XGBoost model
xgb_params <- list(objective = "reg:squarederror", max_depth = 9, eta = 0.3, subsample = 1)
xgb_model <- xgboost(data = xgb_data, params = xgb_params, nrounds = 25)

# Final train-set RMSE: 0.976044 
# > # Train XGBoost model
#> xgb_params <- list(objective = "reg:squarederror", max_depth = 9, eta = 0.3, subsample = 1)
#> xgb_model <- xgboost(data = xgb_data, params = xgb_params, nrounds = 25)
#[1]	train-rmse:2.354883 
#[2]	train-rmse:1.802112 
#[3]	train-rmse:1.453377 
#[4]	train-rmse:1.247689 
#[5]	train-rmse:1.132201 
#[6]	train-rmse:1.068793 
#[7]	train-rmse:1.035379 
#[8]	train-rmse:1.018024 
#[9]	train-rmse:1.008384 
#[10]	train-rmse:1.003118 
#[11]	train-rmse:0.999957 
#[12]	train-rmse:0.995868 
#[13]	train-rmse:0.994121 
#[14]	train-rmse:0.991022 
#[15]	train-rmse:0.987674 
#[16]	train-rmse:0.986393 
#[17]	train-rmse:0.984479 
#[18]	train-rmse:0.983148 
#[19]	train-rmse:0.982165 
#[20]	train-rmse:0.981068 
#[21]	train-rmse:0.980051 
#[22]	train-rmse:0.978978 
#[23]	train-rmse:0.977751 
#[24]	train-rmse:0.976825 
#[25]	train-rmse:0.976044 
# NOTE: The RMSE did not show improvement, and the processing time was around 2 hours
# Code run and tested as of Aug 22, 2023 10:26 am AM

# g.1. Parameter tuning:

#* ### WARNING: running time 55 minutes ###
#* NOTES ABOUT TUNING:
#* increase max_depth to 17 to increase performance
#* reduce nrounds to 15 to reduce processing time
#* keep bootstrapping
#* Start time: 10:45 am
#* End time: 11:40 am

# Prepare data for XGBoost
# Include only the columns used for model training in both train_set and test_data
feature_columns <- setdiff(names(train_set), "rating")
xgb_data <- xgb.DMatrix(
  data = model.matrix(~ . - 1, data = train_set[, feature_columns]), label = train_set$rating)

# Train XGBoost model
xgb_params <- list(objective = "reg:squarederror", max_depth = 17, eta = 0.3, subsample = 1)
xgb_model <- xgboost(data = xgb_data, params = xgb_params, nrounds =15)

# Final train-set RMSE: 0.919032 
#> # Train XGBoost model
#> xgb_params <- list(objective = "reg:squarederror", max_depth = 17, eta = 0.3, subsample = 1)
#> xgb_model <- xgboost(data = xgb_data, params = xgb_params, nrounds =15)
#[1]	train-rmse:2.351116 
#[2]	train-rmse:1.791384 
#[3]	train-rmse:1.434107 
#[4]	train-rmse:1.215202 
#[5]	train-rmse:1.089266 
#[6]	train-rmse:1.019246 
#[7]	train-rmse:0.978951 
#[8]	train-rmse:0.956381 
#[9]	train-rmse:0.943174 
#[10]	train-rmse:0.936961 
#[11]	train-rmse:0.931076 
#[12]	train-rmse:0.927066 
#[13]	train-rmse:0.924460 
#[14]	train-rmse:0.921307 
#[15]	train-rmse:0.919032 
# NOTE: Parameter tuning improves performance and processing time 

print(xgb_model)

# Save Xgboost model
rds_file_path_xgb_model <- file.path(script_directory, "xgb_model.rds")
saveRDS(xgb_model, rds_file_path_xgb_model)

# Prepare test data with the same feature columns
xgb_test_data <- xgb.DMatrix(
  data = model.matrix(~ . - 1, data = test_data[, feature_columns]))

# Make predictions using the XGBoost model on the test data
predicted_ratings <- predict(xgb_model, newdata = xgb_test_data)

# Calculate RMSE
rmse <- RMSE(predicted_ratings, test_data$rating)
print(rmse)
# RMSE: 0.9904257
# NOTE: Perceived over fitting

# Code tested as of Aug 22, 2023 10:42 am with 9 depth and 2 rounds AM
# Code tested as of Aug 22, 2023 11:40 am with 17 depth and 15 rounds AM 

# Plot RMSE gain performance for each model 
# Load libraries
library(ggplot2)

# Arrange data
rmse_values_existing <- c(2.354883, 1.802112, 1.453377, 1.247689, 1.132201, 
                          1.068793, 1.035379, 1.018024, 1.008384, 1.003118, 
                          0.999957, 0.995868, 0.994121, 0.991022, 0.987674, 
                          0.986393, 0.984479, 0.983148, 0.982165, 0.981068, 
                          0.980051, 0.978978, 0.977751, 0.976825, 0.976044)

rmse_values_new <- c(2.351116, 1.791384, 1.434107, 1.215202, 1.089266, 
                     1.019246, 0.978951, 0.956381, 0.943174, 0.936961, 
                     0.931076, 0.927066, 0.924460, 0.921307, 0.919032)

# Make the vectors the same length
max_length <- max(length(rmse_values_existing), length(rmse_values_new))
rmse_values_existing <- c(rmse_values_existing, rep(NA, max_length - length(rmse_values_existing)))
rmse_values_new <- c(rmse_values_new, rep(NA, max_length - length(rmse_values_new)))

# Create a data frame
rmse_df <- data.frame(Existing_RMSE = rmse_values_existing, New_RMSE = rmse_values_new)

# Create a log plot with RMSE gain
plot_3_0 <- ggplot(rmse_df, aes(x = seq_along(Existing_RMSE))) +
  geom_line(aes(y = Existing_RMSE, color = "Existing RMSE")) +
  geom_line(aes(y = New_RMSE, color = "New RMSE")) +
  labs(
    x = "Index", 
    y = "RMSE",
    title = "Xgboost Training: RMSE Gain with Parameter Tuning") +
  scale_color_manual(values = c("Existing RMSE" = "#456990", "New RMSE" = "#F45B69")) +
  theme_economist_white() +
  scale_y_log10()

# Save plots for RDM file
plot_file_path_3_0 <- file.path(script_directory, "RMSE_gain_plot.png")
ggsave(plot_file_path_3_0, plot_3_0, width = 8, height = 6)

# Plot code tested and saved as of Sep 2, 2023 8:43 am AM

# h.1.a. LightGBM model using 500K records
# NOTE: LightGBM is more time efficient, therefore increased number of records to 500K

# Load libraries
if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(lightgbm)) install.packages("lightgbm", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(lightgbm)

# Set a seed
set.seed(1974)

# Create a random sample of maximum 500K records
max_records <- 500000
sampled_edx <- edx %>% sample_n(max_records)

# Save the sampled_edx data frame to a new CSV file
write.csv(sampled_edx, "sampled_edx.csv", row.names = FALSE)

# Create training and test files in the 500-thousand-record file
set.seed(1974)
indexes <- split(1:nrow(sampled_edx), sampled_edx$userId)
test_ind <- sapply(indexes, function(ind) sample(ind, ceiling(length(ind) * 0.2))) |>
  unlist(use.names = TRUE) |> sort()
test_set <- sampled_edx[test_ind,]
train_set <- sampled_edx[-test_ind,]

test_set <- test_set |> 
  semi_join(train_set, by = "movieId")
train_set <- train_set |> 
  semi_join(test_set, by = "movieId")

y <- select(train_set, movieId, userId, rating) |>
  pivot_wider(names_from = movieId, values_from = rating) 
rnames <- y$userId
y <- as.matrix(y[,-1])
rownames(y) <- rnames

movie_map <- train_set |> select(movieId, title) |> distinct(movieId, .keep_all = TRUE)

# Run LightGBM model

# Load libraries
if (!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if (!require(lightgbm)) install.packages("lightgbm", repos = "http://cran.us.r-project.org")
library(lubridate)
library(lightgbm)

# Set seed
set.seed(1974)

# Convert timestamp to datetime and extract day of the week
sampled_edx$timestamp <- as.POSIXlt(sampled_edx$timestamp, origin = "1970-01-01", tz = "UTC")
sampled_edx$day_of_week <- weekdays(sampled_edx$timestamp)

# Calculate mean rating for each day of the week
day_effect <- sampled_edx %>%
  group_by(day_of_week) %>%
  summarize(mean_rating = mean(rating, na.rm = TRUE))

# Calculate the overall mean rating
mu <- mean(sampled_edx$rating, na.rm = TRUE)

# Calculate the effect of the day of the week
day_effect$day_effect <- day_effect$mean_rating - mu

# Calculate movie effects
b_i <- colMeans(y - mu, na.rm = TRUE)

fit_movies <- data.frame(movieId = as.integer(colnames(y)), 
                         mu = mu, b_i = b_i)

# Add day_of_week to train_set test_set
train_set$day_of_week <- weekdays(as.POSIXlt(train_set$timestamp, origin = "1970-01-01", tz = "UTC"))
test_set$day_of_week <- weekdays(as.POSIXlt(test_set$timestamp, origin = "1970-01-01", tz = "UTC"))

# Joining with day effect and movie effect
train_data <- left_join(train_set, fit_movies, by = "movieId") %>%
  left_join(day_effect, by = c("day_of_week" = "day_of_week"))
test_data <- left_join(test_set, fit_movies, by = "movieId") %>%
  left_join(day_effect, by = c("day_of_week" = "day_of_week"))

# Prepare data for LightGBM ### NTR: Should read train_data in this section, apologies :) AM
lgb_data <- lgb.Dataset(
  data = as.matrix(train_data %>% select(-rating)), label = train_data$rating)

# Define LightGBM parameters
lgb_params <- list(objective = "regression", 
                   max_depth = 10, num_leaves = 64, learning_rate = 0.3, metric = "rmse")

# Train LightGBM model
lgb_model <- lgb.train(params = lgb_params, data = lgb_data, nrounds = 400)

print(lgb_model)

# Save LightGBM model
rds_file_path_lgb_model <- file.path(script_directory, "lgb_model.rds")
saveRDS.lgb.Booster(lgb_model, rds_file_path_lgb_model)

# lgb model saved as of Sep 17, 2023 5:46 pm AM

# Extract feature columns from test_data
test_features <- as.matrix(test_data %>% select(-rating))

# Make predictions using the LightGBM model
lgb_prediction <- predict(lgb_model, test_features)

# Calculate RMSE
rmse <- RMSE(lgb_prediction, test_data$rating)
print(rmse)
#RMSE: [1] 0.9553327 (tested as of Sep 17, 2023 5:46 pm AM)

# Plotting performance of the LightGBM model

# i. Actual vs. Predicted plot
plot_data <- data.frame(Actual = test_data$rating, Predicted = lgb_prediction)
ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(x = "Actual Rating", y = "Predicted Rating", title = "Actual vs. Predicted Ratings (LightGBM)")

# ii. Residual plot
residuals <- test_data$rating - lgb_prediction
plot_data <- data.frame(Predicted = lgb_prediction, Residuals = residuals)
ggplot(plot_data, aes(x = Predicted, y = Residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(x = "Predicted Rating", y = "Residuals", title = "Residual Plot (LightGBM)")

# iii. Distribution of Residuals: A symmetric and centered distribution around zero indicates a good model fit.
plot_3 <- ggplot(data.frame(Residuals = residuals), aes(x = Residuals)) +
  geom_histogram(binwidth = 0.5, fill = "#456990", color = "grey") +
  labs(x = "Residuals", y = "Frequency", title = "Distribution of Residuals (LightGBM)") +
  theme_economist_white() +
  coord_cartesian(xlim = c(min(-3), max(3)))+
  theme(
    plot.title = element_text(hjust = 0.5, size = 10)
  )

# Code tested as of Sep 17, 2023 5:46 pm AM
# Warning due to NA values, but does not comprise the performance of the predictive model

# Save plots for RDM file
plot_file_path_3 <- file.path(script_directory, "residuals_plot_LightGBM.png")
ggsave(plot_file_path_3, plot_3, width = 8, height = 6)

# Plot saved as of Sep 17, 2023 5:50 pm AM

# h.1.b. LightGBM model: Hyper parameter tuning and cross-validation

# Load libraries
if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(lightgbm)) install.packages("lightgbm", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(lightgbm)

# Set a seed
set.seed(1974)

# Create a random sample of maximum 500K records
max_records <- 200000
sampled_edx <- edx %>% sample_n(max_records)

# Save the sampled_edx data frame to a new CSV file
write.csv(sampled_edx, "sampled_edx.csv", row.names = FALSE)

# Create training and test files in the 500-thousand-record file
set.seed(1974)
indexes <- split(1:nrow(sampled_edx), sampled_edx$userId)
test_ind <- sapply(indexes, function(ind) sample(ind, ceiling(length(ind) * 0.2))) |>
  unlist(use.names = TRUE) |> sort()
test_set <- sampled_edx[test_ind,]
train_set <- sampled_edx[-test_ind,]

test_set <- test_set |> 
  semi_join(train_set, by = "movieId")
train_set <- train_set |> 
  semi_join(test_set, by = "movieId")

y <- select(train_set, movieId, userId, rating) |>
  pivot_wider(names_from = movieId, values_from = rating) 
rnames <- y$userId
y <- as.matrix(y[,-1])
rownames(y) <- rnames

movie_map <- train_set |> select(movieId, title) |> distinct(movieId, .keep_all = TRUE)

# Run LightGBM model

# Load libraries
if (!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if (!require(lightgbm)) install.packages("lightgbm", repos = "http://cran.us.r-project.org")
library(lubridate)
library(lightgbm)

# Set seed
set.seed(1974)

# Convert timestamp to datetime and extract day of the week
sampled_edx$timestamp <- as.POSIXlt(sampled_edx$timestamp, origin = "1970-01-01", tz = "UTC")
sampled_edx$day_of_week <- weekdays(sampled_edx$timestamp)

# Calculate mean rating for each day of the week
day_effect <- sampled_edx %>%
  group_by(day_of_week) %>%
  summarize(mean_rating = mean(rating, na.rm = TRUE))

# Calculate the overall mean rating
mu <- mean(sampled_edx$rating, na.rm = TRUE)

# Calculate the effect of the day of the week
day_effect$day_effect <- day_effect$mean_rating - mu

# Calculate movie effects
b_i <- colMeans(y - mu, na.rm = TRUE)

fit_movies <- data.frame(movieId = as.integer(colnames(y)), 
                         mu = mu, b_i = b_i)

# Add day_of_week to test_set
train_set$day_of_week <- weekdays(as.POSIXlt(train_set$timestamp, origin = "1970-01-01", tz = "UTC"))
test_set$day_of_week <- weekdays(as.POSIXlt(test_set$timestamp, origin = "1970-01-01", tz = "UTC"))

# Joining with day effect and movie effect
train_data <- left_join(train_set, fit_movies, by = "movieId") %>%
  left_join(day_effect, by = c("day_of_week" = "day_of_week"))
test_data <- left_join(test_set, fit_movies, by = "movieId") %>%
  left_join(day_effect, by = c("day_of_week" = "day_of_week"))

# Prepare data for LightGBM
lgb_data <- lgb.Dataset(
  data = as.matrix(train_data %>% select(-rating)), label = train_data$rating)

# Set seed
set.seed(1974)

# Define LightGBM parameters
lgb_params <- list(objective = "regression", 
                   max_depth = 06, num_leaves = 8, learning_rate = 0.05, metric = "rmse")

# Perform cross-validation
cv_results <- lgb.cv(params = lgb_params, 
                     data = lgb_data, 
                     nrounds = 400, 
                     stratified = FALSE, 
                     nfold = 10,
                     early_stopping_rounds = 10,
                     verbose = 0) 

# Print cross-validation results
print(cv_results)

# Parameter tuning: train LightGBM model
final_lgb_model_cv <- lgb.train(params = lgb_params, data = lgb_data, nrounds = 42)

# Extract feature columns from test_data
test_features <- as.matrix(test_data %>% select(-rating))

# Make predictions using the LightGBM model
lgb_prediction_cv <- predict(final_lgb_model_cv, test_features)

# Calculate RMSE
rmse <- RMSE(lgb_prediction_cv, test_data$rating)
print(rmse)
# RMSE results using differnt parameters
#RMSE: 0.6894698 (tested as of Sep 4, 2023 9:09 pm AM)
#RMSE: 0.897992
#RMSE: 0.9143711
#RMSE: 0.9330465 (nrounds = 50)
#RMSE: 0.9439734 (nrounds = 25)
#RMSE: 0.989045 (nrounds = 25, learning_rate = 0.1)
#RMSE: 0.9636733 (nrounds = 25, learning_rate = 0.05)
#RMSE: 0.9636542 (nrounds = 25, learning_rate = 0.05, max_depth = 08)
#RMSE: 0.9637139 (nrounds = 25, learning_rate = 0.05, max_depth = 07)
#RMSE: 0.9667125 (nrounds = 25, learning_rate = 0.05, max_depth = 07, leaves = 16)
#RMSE: 0.9696775 (nrounds = 25, learning_rate = 0.05, max_depth = 07, leaves = 8)
#RMSE: 0.9687746 (nrounds = 26, learning_rate = 0.05, max_depth = 06, leaves = 8)
#RMSE: 0.9679524 (nrounds = 27, learning_rate = 0.05, max_depth = 06, leaves = 8)
#RMSE: 0.9671818 (nrounds = 28, learning_rate = 0.05, max_depth = 06, leaves = 8)
#RMSE: 0.9664797 (nrounds = 29, learning_rate = 0.05, max_depth = 06, leaves = 8)
#RMSE: 0.9658082 (nrounds = 30, learning_rate = 0.05, max_depth = 06, leaves = 8)
#RMSE: 0.9632374 (nrounds = 35, learning_rate = 0.05, max_depth = 06, leaves = 8)
#RMSE: 0.9614663 (nrounds = 40, learning_rate = 0.05, max_depth = 06, leaves = 8)
#RMSE: 0.9609122 (nrounds = 42, learning_rate = 0.05, max_depth = 06, leaves = 8)
#RMSE: 0.9623453

# Code tested as of Sep 17, 2023 6:08 pm AM

# Save LightGBM_cv model
rds_file_path_lgb_model_cv <- file.path(script_directory, "final_lgb_model_cv.rds")
saveRDS.lgb.Booster(final_lgb_model_cv, rds_file_path_lgb_model_cv)

# lgb_cv odel saved as of Sep 17, 2023 6:09 pm AM

# Extract feature columns from test_data
test_features <- as.matrix(test_data %>% select(-rating))

# Make predictions using the LightGBM model
lgb_cv_prediction <- predict(final_lgb_model_cv, test_features)

# Calculate RMSE
rmse <- RMSE(lgb_cv_prediction, test_data$rating)
print(rmse)
#RMSE: [1] 0.9623453
# Code tested as of Sep 17, 2023 6:09 pm AM

# Plotting performance of the LightGBM model ### NEED TO BE UPDATED to lgb_cv ###

# i. Actual vs. Predicted plot
plot_data <- data.frame(Actual = test_data$rating, Predicted = lgb_cv_prediction)
ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(x = "Actual Rating", y = "Predicted Rating", title = "Actual vs. Predicted Ratings (LightGBM)")

# ii. Residual plot
residuals <- test_data$rating - lgb_cv_prediction
plot_data <- data.frame(Predicted = lgb_cv_prediction, Residuals = residuals)
ggplot(plot_data, aes(x = Predicted, y = Residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(x = "Predicted Rating", y = "Residuals", title = "Residual Plot (LightGBM)")

# iii. Distribution of Residuals: A symmetric and centered distribution around zero indicates a good model fit.
plot_3 <- ggplot(data.frame(Residuals = residuals), aes(x = Residuals)) +
  geom_histogram(binwidth = 0.5, fill = "#456990", color = "grey") +
  labs(x = "Residuals", y = "Frequency", title = "Distribution of Residuals (LightGBM)") +
  theme_economist_white() +
  coord_cartesian(xlim = c(min(-3), max(3)))+
  theme(
    plot.title = element_text(hjust = 0.5, size = 10)
  )

print(plot_3)

# Code tested as of Aug 23, 2023 8:20 am
# Code re-tested as of Sep 17, 2023 6:12 pm AM
# Warning due to NA values, but does not comprise the performance of the predictive model

# Save plots for RDM file
plot_file_path_3 <- file.path(script_directory, "residuals_plot_LightGBM.png")
ggsave(plot_file_path_3, plot_3, width = 8, height = 6)

# Plot saved as of Sep 17, 2023 9:01 pm AM

# i Random Forest

##############################################
### WARNING: Code takes 25 minutes running ###
##############################################

# Load libraries
if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(randomForest)

# Set seed
set.seed(1974)

# Create a random sample of maximum 500K records
max_records <- 100000
sampled_edx <- edx %>% sample_n(max_records)

# Save the sampled_edx data frame to a new CSV file
write.csv(sampled_edx, "sampled_edx.csv", row.names = FALSE)

# Create training and test files in the 100-hundred-record file
set.seed(1974)
indexes <- split(1:nrow(sampled_edx), sampled_edx$userId)
test_ind <- sapply(indexes, function(ind) sample(ind, ceiling(length(ind) * 0.2))) |>
  unlist(use.names = TRUE) |> sort()
test_set <- sampled_edx[test_ind,]
train_set <- sampled_edx[-test_ind,]

test_set <- test_set |> 
  semi_join(train_set, by = "movieId")
train_set <- train_set |> 
  semi_join(test_set, by = "movieId")

y <- select(train_set, movieId, userId, rating) |>
  pivot_wider(names_from = movieId, values_from = rating) 
rnames <- y$userId
y <- as.matrix(y[,-1])
rownames(y) <- rnames

movie_map <- train_set |> select(movieId, title) |> distinct(movieId, .keep_all = TRUE)

# Convert timestamp to datetime and extract day of the week
sampled_edx$timestamp <- as.POSIXlt(sampled_edx$timestamp, origin = "1970-01-01", tz = "UTC")
sampled_edx$day_of_week <- weekdays(sampled_edx$timestamp)

# Calculate average rating for each day of the week
day_effect <- sampled_edx %>%
  group_by(day_of_week) %>%
  summarize(mean_rating = mean(rating, na.rm = TRUE))

# Calculate the overall average rating
mu <- mean(sampled_edx$rating, na.rm = TRUE)

# Calculate movie effects

# a. Movie Effects
b_i <- colMeans(y - mu, na.rm = TRUE)

fit_movies <- data.frame(movieId = as.integer(colnames(y)), 
                         mu = mu, b_i = b_i)

# Calculate the effect of the day of the week
day_effect$day_effect <- day_effect$mean_rating - mu

# Add day_of_week to test_set
train_set$day_of_week <- weekdays(as.POSIXlt(train_set$timestamp, origin = "1970-01-01", tz = "UTC"))
test_set$day_of_week <- weekdays(as.POSIXlt(test_set$timestamp, origin = "1970-01-01", tz = "UTC"))

# Joining with day effect and movie effect
train_data <- left_join(train_set, fit_movies, by = "movieId") %>%
  left_join(day_effect, by = c("day_of_week" = "day_of_week"))
test_data <- left_join(test_set, fit_movies, by = "movieId") %>%
  left_join(day_effect, by = c("day_of_week" = "day_of_week"))

# v2

# Prepare features and labels for training
train_features <- train_set %>% select(-rating)
train_labels <- train_set$rating

# Check for missing values
structure(train_features )
is.na(train_features )
complete.cases(train_features )

# Train Random Forest model

####################################################
#### WARNING: code takes several minutes running ###
####################################################

# Set seed
set.seed(1974)

rf_model <- randomForest(
  x = train_features, 
  y = train_labels, 
  ntree = 100, 
  mtry = sqrt(ncol(train_features)))

print(rf_model)

# Save RF model
rds_file_path_rf_model <- file.path(script_directory, "rf_model.rds")
saveRDS(rf_model, rds_file_path_rf_model)

# RF model saved as of Sep 4, 2023 11:58 pm AM

# Read the RDS file
rds_file_path_rf_model <- file.path(script_directory, "rf_model.rds")
loaded_rf_model <- readRDS(rds_file_path_rf_model)

# Make predictions using the Random Forest model
rf_predictions <- predict(rf_model, newdata = test_data)

# Calculate RMSE
rmse <- sqrt(mean((rf_predictions - test_data$rating)^2))
print(rmse)

#RMSE 0.5400452
#RMSE 0.9875306 (Sep 4, 2023)
#RMSE 1.025289 (Sep 5, 2023 6:01 pm)
#NOTE: Training the RF model takes approx. 25 minutes

#v2

# Plotting performance of the Random Forest model

# iv. Distributions of residuals
plot_4 <- ggplot(data.frame(Residuals = residuals_rf), aes(x = Residuals)) +
  geom_histogram(binwidth = 0.5, fill = "#456990", color = "grey") +
  theme_economist_white() +
  labs(
    x = "Residuals", 
    y = "Frequency", 
    title = "Distribution of Residuals (Random Forest)") +
  coord_cartesian(xlim = c(min(-3), max(3)))+
  theme(
    plot.title = element_text(hjust = 0.5, size = 10)
  )

print(plot_4)

summary(residuals_rf)

# Code tested as of Aug 27, 2023 7:25 am AM

# Save plots for RDM file
plot_file_path_4 <- file.path(script_directory, "residuals_plot_Random_Forest.png")
ggsave(plot_file_path_4, plot_4, width = 8, height = 6) 

# Plot saved as of Aug 27, 2023 7:33 pm AM

# Code reviewed as of Sep 2, 2023 7:44 pm AM

# i.b Random Forest Cross-validation
# Create a RF CV model Sep 5, 2023

# Load libraries
if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if (!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(randomForest)
library(caret)

# Set seed
set.seed(1974)

# Create a random sample of maximum 1MM records
max_records <- 1000000
sampled_edx <- edx %>% sample_n(max_records)

# Save the sampled_edx data frame to a new CSV file
write.csv(sampled_edx, "sampled_edx.csv", row.names = FALSE)

# Read the CSV file into a data frame
sampled_edx <- read.csv("sampled_edx.csv")

# Check the structure of the data frame
str(sampled_edx)

# Create training and test files in the 100-hundred-record file
set.seed(1974)
indexes <- split(1:nrow(sampled_edx), sampled_edx$userId)
test_ind <- sapply(indexes, function(ind) sample(ind, ceiling(length(ind) * 0.2))) |>
  unlist(use.names = TRUE) |> sort()
test_set <- sampled_edx[test_ind,]
train_set <- sampled_edx[-test_ind,]

test_set <- test_set |> 
  semi_join(train_set, by = "movieId")
train_set <- train_set |> 
  semi_join(test_set, by = "movieId")

y <- select(train_set, movieId, userId, rating) |>
  pivot_wider(names_from = movieId, values_from = rating) 
rnames <- y$userId
y <- as.matrix(y[,-1])
rownames(y) <- rnames

movie_map <- train_set |> select(movieId, title) |> distinct(movieId, .keep_all = TRUE)

# Convert timestamp to datetime and extract day of the week
sampled_edx$timestamp <- as.POSIXlt(sampled_edx$timestamp, origin = "1970-01-01", tz = "UTC")
sampled_edx$day_of_week <- weekdays(sampled_edx$timestamp)

# Calculate average rating for each day of the week
day_effect <- sampled_edx %>%
  group_by(day_of_week) %>%
  summarize(mean_rating = mean(rating, na.rm = TRUE))

# Calculate the overall average rating
mu <- mean(sampled_edx$rating, na.rm = TRUE)

# Calculate movie effects

# a. Movie Effects
b_i <- colMeans(y - mu, na.rm = TRUE)

fit_movies <- data.frame(movieId = as.integer(colnames(y)), 
                         mu = mu, b_i = b_i)

# Calculate the effect of the day of the week
day_effect$day_effect <- day_effect$mean_rating - mu

# Add day_of_week to train_set and test_set
train_set$day_of_week <- weekdays(as.POSIXlt(train_set$timestamp, origin = "1970-01-01", tz = "UTC"))
test_set$day_of_week <- weekdays(as.POSIXlt(test_set$timestamp, origin = "1970-01-01", tz = "UTC"))

# Joining with day effect and movie effect
train_data <- left_join(train_set, fit_movies, by = "movieId") %>%
  left_join(day_effect, by = c("day_of_week" = "day_of_week"))
test_data <- left_join(test_set, fit_movies, by = "movieId") %>%
  left_join(day_effect, by = c("day_of_week" = "day_of_week"))

# Prepare features and labels for training
train_features <- train_set %>% select(-rating)
train_labels <- train_set$rating

# Check for missing values
structure(train_features )
is.na(train_features )
complete.cases(train_features )

# Define the Random Forest model
rf_model <- randomForest(
  x = train_features, 
  y = train_labels, 
  ntree = 5, 
  mtry = sqrt(ncol(train_features)),
  importance = TRUE
)

# Perform k-fold cross-validation
num_folds <- 5
rf_cv <- train(
  x = train_features, 
  y = train_labels,
  method = "rf",
  trControl = trainControl(method = "cv", number = num_folds),
  tuneGrid = expand.grid(.mtry = sqrt(ncol(train_features))),
  importance = TRUE
)

# Print cross-validation results
print(rf_cv)

# Get the best model from cross-validation
best_rf_model <- rf_cv$finalModel

# Make predictions using the best model
rf_predictions <- predict(best_rf_model, newdata = test_data)

# Calculate RMSE
rmse_rf_cv <- sqrt(mean((rf_predictions - test_data$rating)^2))
print(rmse_rf_cv)
#RMSE: 0.9690424

# Code tested as of Sep 5, 2023 9:44 pm AM

# Save rf_cv_model
rf_model_file_path_cv <- file.path(script_directory, "best_rf_model_cv.rds")
saveRDS(best_rf_model, rf_model_file_path_cv)

# best_rf_model saved as of sep 5, 2023 9:46 pm AM

# Plotting performance of the Random Forest Cross-Validation model

# Load library
if (!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
library(ggplot2)

# i. Actual vs. Predicted plot
plot_data_rf <- data.frame(Actual = test_data$rating, Predicted = rf_predictions)
ggplot(plot_data_rf, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(x = "Actual Rating", 
       y = "Predicted Rating", 
       title = "Actual vs. Predicted Ratings (Random Forest Cross-Validation)")

# ii. Residual plot
residuals_rf <- test_data$rating - rf_predictions
plot_data_rf <- data.frame(Predicted = rf_predictions, Residuals = residuals_rf)
ggplot(plot_data_rf, aes(x = Predicted, y = Residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(x = "Predicted Rating", 
       y = "Residuals", 
       title = "Residual Plot (Random Forest Cross-Validation)")

# iii. Distributions of residuals
plot_5 <- ggplot(data.frame(Residuals = residuals_rf), aes(x = Residuals)) +
  geom_histogram(binwidth = 0.5, fill = "#456990", color = "grey") +
  theme_minimal() +
  labs(
    x = "Residuals", 
    y = "Frequency", 
    title = "Distribution of Residuals (Random Forest Cross-Validation)") +
  coord_cartesian(xlim = c(min(-3), max(3))) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 12)
  )

# Save plots for RDM file
plot_file_path_5 <- file.path(script_directory, "residuals_plot_Random_Forest_CV.png")
ggsave(plot_file_path_5, plot_5, width = 8, height = 6) 

# plot_5 saved as of Sept 5, 2023 9:59 pm AM

# Code tested as of Sep 5, 2023 10:02 pm AM

### Sep 6, 2023, Exploring Genre bias
# z. Genre effect
### Sep 7, 2023, Including Genre effect
#v4
# Load libraries
if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if (!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if (!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if (!require(lightgbm)) install.packages("lightgbm", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(randomForest)
library(caret)
library(lubridate)
library(lightgbm)

# Set a seed
set.seed(1974)

# Load edx data set
edx <- readRDS(rds_file_path_edx)

names(edx)

# Create a random sample of maximum 500K records
max_records <- 500000
sampled_edx <- edx %>% sample_n(max_records)

# Save the sampled_edx data frame to a new CSV file
write.csv(sampled_edx, "sampled_edx.csv", row.names = FALSE)

# Create training and test files in the 500-thousand-record file
set.seed(1974)
indexes <- split(1:nrow(sampled_edx), sampled_edx$userId)
test_ind <- sapply(indexes, function(ind) sample(ind, ceiling(length(ind) * 0.2))) |>
  unlist(use.names = TRUE) |> sort()
test_set <- sampled_edx[test_ind,]
train_set <- sampled_edx[-test_ind,]

# Load library
library(dplyr)

test_set <- test_set |> 
  semi_join(train_set, by = "movieId")
train_set <- train_set |> 
  semi_join(test_set, by = "movieId")

y <- select(train_set, movieId, userId, rating) |>
  pivot_wider(names_from = movieId, values_from = rating) 
rnames <- y$userId
y <- as.matrix(y[,-1])
rownames(y) <- rnames

movie_map <- train_set |> select(movieId, title) |> distinct(movieId, .keep_all = TRUE)

# Concatenate genres and remove '|'
sampled_edx <- sampled_edx %>%
  mutate(NewGenre = str_replace_all(genres, "\\|", ""))

colnames(sampled_edx)

# Calculate the average rating per NewGenre
genre_avg_ratings <- sampled_edx %>%
  group_by(NewGenre) %>%
  summarize(AvgRating = mean(rating, na.rm = TRUE))

count(genre_avg_ratings)

# Calculate the overall mean rating
mu <- mean(sampled_edx$rating, na.rm = TRUE)

# Calculate the effect of each genre (b_g)
genre_effect <- genre_avg_ratings %>%
  mutate(b_g = AvgRating - mu) %>%
  select(NewGenre, b_g)

# Order the data frame by b_g in descending order
genre_effect <- genre_effect %>%
  arrange(desc(b_g))

genre_effect

# Save the table to use in RMD
csv_file_path_2 <- file.path(script_directory, "genre_effect.csv")
write.csv(file = csv_file_path_2, genre_effect, row.names = FALSE)

# Load libraries
library(ggplot2)
library(ggthemes)

# Create the plot of genre effect with dots in ascending order
plot_5_0 <- ggplot(genre_effect, aes(x = reorder(NewGenre, b_g), y = b_g)) +
  geom_point() +  # Scatter plot
  labs(x = "Movie Genre", y = "Average Rating Effect") +
  theme_economist_white() +
  theme(axis.text.x = element_blank())  # Omit genre names on the x-axis

# Print the plot
print(plot_5_0)

# Save plots for RDM file
plot_file_path_5_0 <- file.path(script_directory, "scatter_plot_LGB_genre_ratings.png")
ggsave(plot_file_path_5_0, plot_5_0, width = 8, height = 6)

# Plot saved as of Sep 17, 2023 9:55 pm AM

# Data preparation

# Load libraries
if (!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if (!require(lightgbm)) install.packages("lightgbm", repos = "http://cran.us.r-project.org")
library(lubridate)
library(lightgbm)

# Set a seed
set.seed(1974)

# Convert timestamp to datetime and extract day of the week
sampled_edx$timestamp <- as.POSIXlt(sampled_edx$timestamp, origin = "1970-01-01", tz = "UTC")
sampled_edx$day_of_week <- weekdays(sampled_edx$timestamp)

# Calculate mean rating for each day of the week
day_effect <- sampled_edx %>%
  group_by(day_of_week) %>%
  summarize(mean_rating = mean(rating, na.rm = TRUE))

# Calculate the effect of the day of the week
day_effect$day_effect <- day_effect$mean_rating - mu

# Calculate movie effects
b_i <- colMeans(y - mu, na.rm = TRUE)

fit_movies <- data.frame(movieId = as.integer(colnames(y)), 
                         mu = mu, b_i = b_i)

# Add day_of_week to test_set
test_set$day_of_week <- 
  weekdays(as.POSIXlt(test_set$timestamp, origin = "1970-01-01", tz = "UTC"))
train_set$day_of_week <- 
  weekdays(as.POSIXlt(train_set$timestamp, origin = "1970-01-01", tz = "UTC"))

# Joining with day effect and movie effect
test_data <- left_join(test_set, fit_movies, by = "movieId") %>%
  left_join(day_effect, by = c("day_of_week" = "day_of_week"))
train_data <- left_join(train_set, fit_movies, by = "movieId") %>%
  left_join(day_effect, by = c("day_of_week" = "day_of_week"))

# Add NewGenre column to train_set and test_set
train_set <- train_set %>%
  mutate(NewGenre = str_replace_all(genres, "\\|", ""))
test_set <- test_set %>%
  mutate(NewGenre = str_replace_all(genres, "\\|", ""))

# Add genre_effect b_g to train_set and test_set
train_set <- left_join(train_set, genre_effect, by = "NewGenre")
test_set <- left_join(test_set, genre_effect, by = "NewGenre")

# Joining with day effect and movie effect
test_data <- left_join(test_set, genre_effect, by = "NewGenre")
train_data <- left_join(train_set, genre_effect, by = "NewGenre")

# NOTE: Column names of test_set and train_set are equal. Sep 10, 2023 2:58 pm AM

# Model training

# Prepare data for LightGBM
lgb_data <- lgb.Dataset(data = as.matrix(train_data %>% select(-rating)), label = train_data$rating)

# Define LightGBM parameters
lgb_params <- list(objective = "regression", 
                   max_depth = 15, num_leaves = 128, learning_rate = 0.05, metric = "rmse")

# Train LightGBM model
lgb_model <- lgb.train(params = lgb_params, data = lgb_data, nrounds = 400)

# Save LightGBM model
rds_file_path_lgb_model <- file.path(script_directory, "lgb_model.rds")
saveRDS.lgb.Booster(lgb_model, rds_file_path_lgb_model)

# Model testing

# Extract feature columns from test_data
test_features <- as.matrix(test_data %>% select(-rating))

# Make predictions using the LightGBM model
lgb_prediction <- predict(lgb_model, test_features)

# Calculate RMSE
rmse <- RMSE(lgb_prediction, test_data$rating)
print(rmse)
#RMSE: 0.8171995
#RMSE: 0.928528 (num_leaves = 16)
#RMSE: 0.82678 (max_depth = 10, num_leaves = 64, learning_rate = 0.3)
# Code tested as of Sep 7, 2023 11:06 PM AM

#RMSE: 0.9778914
#RMSE: 0.9740821 (learning_rate 0.1)
#RMSE: 0.9744805 (max_depth 20)
#RMSE: 0.9740797 (num_leaves 128)
#RMSE: 0.9738635 (max_depth = 100, num_leaves = 560, learning_rate = 0.005)
#RMSE: 0.9741282 (max_depth = 15, num_leaves = 128, learning_rate = 0.05)
#RMSE: 0.9685876 ( + day-of-the-week)
# Model saved Sep 10, 2023 3:00 pm
# Code tested as of Sep 10, 2023 3:00 pm AM # Train and Test set fixed

# Create plots for Model z

# Create a dataframe for Actual vs. Predicted plot
plot_data_lgb <- data.frame(Actual = test_data$rating, Predicted = lgb_prediction)

# Create the Actual vs. Predicted plot for LightGBM
ggplot(plot_data_lgb, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(x = "Actual Rating", 
       y = "Predicted Rating", 
       title = "Actual vs. Predicted Ratings (LightGBM Plus)")

library(ggplot2)

# Create separate data frames for actual and predicted ratings
actual_data <- data.frame(Rating = plot_data_lgb$Actual, Type = "Actual")
predicted_data <- data.frame(Rating = plot_data_lgb$Predicted, Type = "Predicted")

# Combine the data frames
combined_data <- rbind(actual_data, predicted_data)

# Load library
library(gridExtra)

# Create separate histograms for actual and predicted ratings
hist_actual <- ggplot(actual_data, aes(x = Rating)) +
  geom_histogram(binwidth = 1, fill = "#456990") +
  labs( y = "Frequency", title = "Actual Ratings") +
  theme_economist_white()

hist_predicted <- ggplot(predicted_data, aes(x = Rating)) +
  geom_histogram(binwidth = 1, fill = "#028090") +
  labs( y = "Frequency", title = "Predicted Ratings") +
  theme_economist_white() +
  scale_x_continuous(breaks = 1:5, limits = c(1, 5))

# Arrange the histograms side by side
arranged_plots_0_lgb  <- grid.arrange(hist_actual, hist_predicted, ncol = 2)

# Save the matrix as an image file
matrix_file_path_0_lgb <- 
  file.path(script_directory, "Distribution of Actuals vs Predicted Ratings.png")
ggsave(matrix_file_path_0_lgb, arranged_plots_0_lgb, width = 8, height = 6)

# Calculate residuals for LightGBM
residuals_lgb <- test_data$rating - lgb_prediction

# Create a dataframe for the Residual plot
plot_data_lgb <- data.frame(Predicted = lgb_prediction, Residuals = residuals_lgb)

# Create the Residual plot for LightGBM
ggplot(plot_data_lgb, aes(x = Predicted, y = Residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(x = "Predicted Rating", y = "Residuals", title = "Residual Plot (LightGBM)")

# Create a histogram of residuals for LightGBM
plot_4_0_lgb <- ggplot(data.frame(Residuals = residuals_lgb), aes(x = Residuals)) +
  geom_histogram(binwidth = 0.5, fill = "#456990", color = "grey") +
  theme_economist_white() +
  labs(
    x = "Residuals", 
    y = "Frequency", 
    title = "Distribution of Residuals (LightGBM Plus)"
  ) +
  coord_cartesian(xlim = c(min(-3), max(3))) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 10)
  )

print(plot_4_0_lgb)

# Save the histogram plot for LightGBM residuals
plot_file_path_4_0_lgb <- file.path(script_directory, "residuals_plot_LightGBM_Plus.png")
ggsave(plot_file_path_4_0_lgb, plot_4_0_lgb, width = 8, height = 6)

# Plot saved as of Sep 12, 2023 9:14 pm AM
# Plot re-saved as of Sep 17, 2023 10:03 pm AM
# Code tested as of Sep 12, 2023 9:15 pm AM
# Code improved and re-tested as of Sep 17, 2023 10:03 pm AM

# Create a matrix with the distribution of residuals for the LGBM, RF and RFCV models

# Load the required packages
library(gridExtra)
library(png)

# Define the path to the PNG files
script_directory <- dirname(rstudioapi::getActiveDocumentContext()$path)
png_file_path_a <- file.path(script_directory, "residuals_plot_LightGBM.png")
png_file_path_b <- file.path(script_directory, "residuals_plot_Random_Forest.png")
png_file_path_c <- file.path(script_directory, "residuals_plot_Random_Forest_CV.png")
png_file_path_d <- file.path(script_directory, "residuals_plot_LightGBM_Plus.png")

# Read the PNG files as images
png_image_a <- readPNG(png_file_path_a)
png_image_b <- readPNG(png_file_path_b)
png_image_c <- readPNG(png_file_path_c)
png_image_d <- readPNG(png_file_path_d)

# Convert the PNG images into plot objects
plot_a <- rasterGrob(png_image_a)
plot_b <- rasterGrob(png_image_b)
plot_c <- rasterGrob(png_image_c)
plot_d <- rasterGrob(png_image_d)

# Arrange the plots in a 2x2 grid
arranged_plots <- grid.arrange(plot_a, plot_b, plot_c, plot_d, ncol = 2, nrow = 2)

# Display the arranged plots
arranged_plots

# Save the matrix as an image file
matrix_file_path_1 <- file.path(script_directory, "Distribution of Residuals LGBM RF RF_CV and LGBM_Plus.png")
ggsave(matrix_file_path_1, arranged_plots, width = 10, height = 6)

# Matrix plot saved as of Aug 31, 2023 8:01 pm AM
# Matrix 2x2 saved as of Sep 18, 2023 10:25 pm AM

# Code reviewed

#***********************************************************************
#* 3. Model Validation with hold-out test
#***********************************************************************

# 3.1. Validating LightGBM model performance

### NEW code to validate lgb model ### Sep 4 3:17 pm

# Load libraries
if (!require(lightgbm)) install.packages("lightgbm", repos = "http://cran.us.r-project.org")
library(lightgbm)

# Load LightGBM model
rds_file_path_lgb_model <- file.path(script_directory, "lgb_model.rds")
lgb_model <- readRDS.lgb.Booster(rds_file_path_lgb_model)

print(lgb_model)

# Read RDS file
rds_file_path_final <- file.path(script_directory, "final_holdout_test_data.rds")
final_holdout_test <- readRDS(rds_file_path_final)

# Preprocess the final_holdout_test dataset (similar to the training data preprocessing)
final_holdout_test$timestamp <- as.POSIXlt(final_holdout_test$timestamp, origin = "1970-01-01", tz = "UTC")
final_holdout_test$day_of_week <- weekdays(final_holdout_test$timestamp)

# Join with movie effects and day effects
final_holdout_test <- left_join(final_holdout_test, fit_movies, by = "movieId") %>%
  left_join(day_effect, by = c("day_of_week" = "day_of_week"))

# Check for missing values
print(sum(is.na(final_holdout_test)))

# Remove rows with empty data in any column
final_holdout_test <- na.omit(final_holdout_test)

# Check for missing values
print(sum(is.na(final_holdout_test)))

# Check each row for missing values and sum the result
num_rows_with_empty_data <- sum(apply(final_holdout_test, 1, function(row) any(is.na(row))))

# Print the result
print(num_rows_with_empty_data)

# Prepare data for LightGBM (excluding 'rating' column)
lgb_test_data <- lgb.Dataset(data = as.matrix(final_holdout_test %>% select_if(is.numeric) %>% select(-rating)), params = lgb_params)

# Prepare data for LightGBM
lgb_test_data <- lgb.Dataset(data = as.matrix(final_holdout_test %>% select_if(is.numeric)), params = lgb_params)

# Extract feature columns from test_data
final_holdout_test_features <- as.matrix(final_holdout_test %>% select(-rating))

# Set seed
set.seed(1974)

# Make predictions using the LightGBM model
lgb_prediction_holdout_test <- predict(final_lgb_model_cv, final_holdout_test_features)

# Calculate RMSE
rmse_final <- RMSE(lgb_prediction_holdout_test, final_holdout_test$rating)
print(rmse_final)
#RMSE: 1.269605 ### DATA TO BUILD PLOT ### (tested as of Sep 4, 2023 7:01 pm AM)
#RMSE: 1.108707
#RMSE: 1.087124
#RMSE: 1.068618
#RMSE: 1.027678
#RMSE: 0.989045
#RMSE: 0.9807888
#RMSE: 0.9803929
#RMSE: 0.9803435
#RMSE: 0.970585
#RMSE: 0.9669355
#RMSE: 0.9661415
#RMSE: 0.9656078
#RMSE: 0.9646446
#RMSE: 0.9641252
#RMSE: 0.9633893
#RMSE: 0.961221
#RMSE: 0.9597464
#RMSE: 0.9593755

# Plot performance

# Load libraries
library(ggthemes)

# a. Actual vs. Predicted plot:

actual_ratings <- final_holdout_test$rating

# Create a data frame for plotting
plot_data <- data.frame(Actual = actual_ratings, Predicted = lgb_prediction_holdout_test)

# Scatter plot of actual vs. predicted ratings
library(ggplot2)
ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Actual vs. Predicted Ratings",
       x = "Actual Ratings",
       y = "Predicted Ratings") +
  theme_minimal()

# b. Plot Residuals

# Calculate residuals
residuals <- actual_ratings - lgb_prediction_holdout_test

# Create a data frame for plotting
residual_plot_data <- data.frame(Actual = actual_ratings, Residual = residuals)

# Scatter plot of residuals vs. actual ratings
library(ggplot2)
ggplot(residual_plot_data, aes(x = Actual, y = Residual)) +
  geom_point(color = "steelblue") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residuals Plot",
       x = "Actual Ratings",
       y = "Residuals") +
  theme_minimal()

# c. Distribition of Residuals

# Calculate residuals
residuals <- actual_ratings - lgb_prediction_holdout_test

# Create a histogram of residuals
library(ggplot2)
plot_6 <- ggplot(data.frame(Residuals = residuals), aes(x = Residuals)) +
  geom_histogram(binwidth = 0.5, fill = "#456990", color = "grey") +
  labs(title = "Distribution of Residuals Model Validation LightGBM",
       x = "Residuals",
       y = "Frequency") +
  theme_economist_white()

# Save plots for RDM file
plot_file_path_6 <- file.path(script_directory, "residuals_plot_LightGBM_Validation.png")
ggsave(plot_file_path_6, plot_6, width = 8, height = 6)

# Plot tested as of Sept 4, 2023 11:03 pm AM

# Alternative distribution of residuals plot

# Create a density plot of residuals
ggplot(data.frame(Residuals = residuals), aes(x = Residuals)) +
  geom_density(fill = "#456990", color = "grey") +
  labs(title = "Distribution of Residuals Model Validation LightGBM",
       x = "Residuals",
       y = "Density") +
  theme_economist_white()

# Additional code tested as of Aug 25, 1:29 pm AM

# Calculating KS for LightGBM model

# Load libraries
library(lightgbm)
library(dplyr)
library(data.table)

# Set seed
set.seed(1974)

# Add day_of_week to final_holdout_test
final_holdout_test$day_of_week <- weekdays(as.POSIXlt(final_holdout_test$timestamp, origin = "1970-01-01", tz = "UTC"))

# Joining with day effect and movie effect
test_data_final <- left_join(final_holdout_test, fit_movies, by = "movieId") %>%
  left_join(day_effect, by = c("day_of_week" = "day_of_week"))

# Prepare data for LightGBM
lgb_data_final <- lgb.Dataset(data = as.matrix(test_data_final %>% select(-rating)), label = test_data_final$rating)

# Make predictions using the LightGBM model
lgb_prediction_final <- predict(lgb_model, as.matrix(test_data_final %>% select(-rating)))

# Calculate the KS statistic
ks_statistic <- ks.test(lgb_prediction_holdout_test, final_holdout_test$rating)$statistic
print(ks_statistic)

#> print(ks_statistic)
#D 
#0.3751627 (Sep 4, 2023 11:06 pm AM)

# Plot KS

library(ggplot2)

# Create data frames for predicted and actual ratings
cdf_data <- data.frame(
  Predicted = sort(lgb_prediction_holdout_test),
  Actual = sort(final_holdout_test$rating)
)

# Create the CDF plot
plot_7 <- ggplot(cdf_data, aes(x = Predicted)) +
  stat_ecdf(aes(color = "Predicted"), geom = "step") +
  geom_step(data = cdf_data, aes(x = Actual, color = "Actual"), stat = "ecdf") +
  labs(title = "CDF of Predicted vs Actual Ratings Light GBM Validation",
       x = "Rating",
       y = "Cumulative Probability") +
  theme_economist_white() +
  theme(legend.position = "top") +
  scale_color_manual(values = c("Predicted" = "#456990", "Actual" = "#F45B69")) +
  annotate("text", x = 1.5, y = 0.2, label = paste("KS Statistic =", round(ks_statistic, 4)), color = "black") +
  scale_x_continuous(limits = c(0, 5))

# Save plots for RDM file
plot_file_path_7 <- file.path(script_directory, "ks_plot_LightGBM_Validation.png")
ggsave(plot_file_path_7, plot_7, width = 8, height = 6)

# KS plot saved as of Sep 4, 2023 11:09 pm AM

# Code updated and tested as of Sep 4, 2023 11:09 pm AM

# 3.2.a. Validating Random Forest RF model performance

#v2 ### WORKS as of Sep 5, 2023 ###
# i Random Forest validation

# Load libraries
if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(randomForest)
library(dplyr)

# Load final_holdout_test data
final_holdout_test <- readRDS(rds_file_path_final)

# Load the rf_model
rf_model_file_path <- file.path(script_directory, "rf_model.rds")
rf_model <- readRDS(rf_model_file_path)

# Transform final_holdout_test dataset

# Convert timestamp to datetime and extract day of the week
final_holdout_test$timestamp <- as.POSIXlt(final_holdout_test$timestamp, origin = "1970-01-01", tz = "UTC")
final_holdout_test$day_of_week <- weekdays(final_holdout_test$timestamp)

# Calculate mean rating for each day of the week
day_effect <- final_holdout_test %>%
  group_by(day_of_week) %>%
  summarize(mean_rating = mean(rating, na.rm = TRUE))

# Calculate the overall mean rating
mu <- mean(final_holdout_test$rating, na.rm = TRUE)

# Calculate the effect of the day of the week
day_effect$day_effect <- day_effect$mean_rating - mu

# Calculate movie effects
b_i <- colMeans(y - mu, na.rm = TRUE)

fit_movies <- data.frame(movieId = as.integer(colnames(y)), 
                         mu = mu, b_i = b_i)

# Add day_of_week to test_set
final_holdout_test$day_of_week <- 
  weekdays(as.POSIXlt(final_holdout_test$timestamp, origin = "1970-01-01", tz = "UTC"))

# Joining with day effect and movie effect
test_data <- left_join(final_holdout_test, fit_movies, by = "movieId") %>%
  left_join(day_effect, by = c("day_of_week" = "day_of_week"))

# Remove rows with empty data in any column
test_data <- na.omit(test_data)

# Check each row for missing values and sum the result
num_rows_with_empty_data <- sum(apply(test_data, 1, function(row) any(is.na(row))))

# Print the result
print(num_rows_with_empty_data)

# Prepare features and labels
features <- test_data %>% select(-rating)
labels <- test_data$rating

# Make predictions using the Random Forest model
rf_test_predictions <- predict(rf_model, newdata = features)

# Extract the actual ratings from the test_data dataset
test_labels <- test_data$rating

# Ensure lenght of predictions and labels match
if (length(rf_test_predictions) != length(test_labels)) {
  stop("Predictions and labels have different lengths")
}

# Calculate RMSE for the predictions
rmse_test <- sqrt(mean((rf_test_predictions - test_labels)^2, na.rm = TRUE))
print(rmse_test)

# print(rmse_test)
# 0.6207405
# Code tested as of Aug 27, 2023 7:49 pm AM
# NOTE: Code crashed - after repair, RMSE is too high
# print(rmse_test)
# [1] 1.053673
# Code tested as of Sep 4, 2023 11:12 pm AM

# Calculate the KS statistic
ks_statistic <- ks.test(rf_test_predictions, test_data$rating)$statistic
print(ks_statistic)

#> print(ks_statistic)
#D 
#0.2812148 
#print(ks_statistic)
#D 
#0.4791002 (Sep 4, 2023 11:13 pm AM) 

# Plot KS

library(ggplot2)

# Create data frame for predicted and actual ratings
cdf_data_rf <- data.frame(
  Predicted = sort(rf_test_predictions),
  Actual = sort(test_data$rating)
)

# Create CDF plot
plot_8 <- ggplot(cdf_data_rf, aes(x = Predicted)) +
  stat_ecdf(aes(color = "Predicted"), geom = "step") +
  geom_step(data = cdf_data, aes(x = Actual, color = "Actual"), stat = "ecdf") +
  labs(title = "CDF of Predicted vs Actual Ratings (RF model)",
       x = "Rating",
       y = "Cumulative Probability") +
  theme_economist_white() +
  theme(legend.position = "top") +
  scale_color_manual(values = c("Predicted" = "#456990", "Actual" = "#F45B69")) +
  annotate("text", x = 1.5, y = 0.2, label = paste("KS Statistic =", round(ks_statistic, 4)), color = "black") +
  scale_x_continuous(limits = c(0, 5))

# Save plots for RDM file
plot_file_path_8 <- file.path(script_directory, "ks_plot_RF_Validation.png")
ggsave(plot_file_path_8, plot_8, width = 8, height = 6)

# Plot saved as of Aug 28, 2023 1051 pm AM
# New plot saved as of Sep 4, 2023 11:14 pm AM

# 3.1.b. Validating Random Forest best model (CV) performance

# Load libraries
if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(randomForest)
library(dplyr)

# Load final_holdout_test data
final_holdout_test <- readRDS(rds_file_path_final)

# Load the best_rf_model
best_rf_model_file_path <- file.path(script_directory, "best_rf_model_cv.rds")
best_rf_model <- readRDS(best_rf_model_file_path)

# Transform final_holdout_test dataset

# Convert timestamp to datetime and extract day of the week
final_holdout_test$timestamp <- as.POSIXlt(final_holdout_test$timestamp, origin = "1970-01-01", tz = "UTC")
final_holdout_test$day_of_week <- weekdays(final_holdout_test$timestamp)

# Calculate mean rating for each day of the week
day_effect <- final_holdout_test %>%
  group_by(day_of_week) %>%
  summarize(mean_rating = mean(rating, na.rm = TRUE))

# Calculate the overall mean rating
mu <- mean(final_holdout_test$rating, na.rm = TRUE)

# Calculate the effect of the day of the week
day_effect$day_effect <- day_effect$mean_rating - mu

# Calculate movie effects
b_i <- colMeans(y - mu, na.rm = TRUE)

fit_movies <- data.frame(movieId = as.integer(colnames(y)), 
                         mu = mu, b_i = b_i)

# Add day_of_week to test_set
final_holdout_test$day_of_week <- 
  weekdays(as.POSIXlt(final_holdout_test$timestamp, origin = "1970-01-01", tz = "UTC"))

# Joining with day effect and movie effect
test_data <- left_join(final_holdout_test, fit_movies, by = "movieId") %>%
  left_join(day_effect, by = c("day_of_week" = "day_of_week"))

# Remove rows with empty data in any column
test_data <- na.omit(test_data)

# Check each row for missing values and sum the result
num_rows_with_empty_data <- sum(apply(test_data, 1, function(row) any(is.na(row))))

# Print the result
print(num_rows_with_empty_data)

# Prepare features and labels
features <- test_data %>% select(-rating)
labels <- test_data$rating

# Make predictions using the Best Random Forest model
rf_test_predictions <- predict(best_rf_model, newdata = features)

# Extract the actual ratings from the test_data dataset
test_labels <- test_data$rating

# Ensure the dimensions of predictions and labels match
if (length(rf_test_predictions) != length(test_labels)) {
  stop("Predictions and labels have different lengths")
}

# Calculate RMSE for the predictions
rmse_test <- sqrt(mean((rf_test_predictions - test_labels)^2, na.rm = TRUE))

# Print RMSE
print(rmse_test)
# RMSE: 1.149998

# Calculate the KS statistic
ks_statistic <- ks.test(rf_test_predictions, test_data$rating)$statistic
print(ks_statistic)

#D 
#0.5402299

# Plot KS
library(ggplot2)

# Create data frame for predicted and actual ratings
cdf_data_rf <- data.frame(
  Predicted = sort(rf_test_predictions),
  Actual = sort(test_data$rating)
)

# Create CDF plot
plot_9 <- ggplot(cdf_data_rf, aes(x = Predicted)) +
  stat_ecdf(aes(color = "Predicted"), geom = "step") +
  geom_step(data = cdf_data_rf, aes(x = Actual, color = "Actual"), stat = "ecdf") +
  labs(title = "CDF of Predicted vs Actual Ratings (Best RF model)",
       x = "Rating",
       y = "Cumulative Probability") +
  theme_economist_white() +
  theme(legend.position = "top") +
  scale_color_manual(values = c("Predicted" = "#456990", "Actual" = "#F45B69")) +
  annotate("text", x = 1.5, y = 0.2, label = paste("KS Statistic =", round(ks_statistic, 4)), color = "black") +
  scale_x_continuous(limits = c(0, 5))

# Save plots for RDM file
plot_file_path_9 <- file.path(script_directory, "ks_plot_Best_RF_Validation.png")
ggsave(plot_file_path_9, plot_9, width = 8, height = 6)

# KS plot for best_rf_cv model saved as of Sep 13, 2023 9:56 pm AM
# Code tested as of Sep 5, 2023 10:33 pm AM

# Code ready for final review as of Sep 5, 2023 10:59 pm AM

# Model Validation model z. lgb model with b_g

#v4
# Load libraries
if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if (!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if (!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if (!require(lightgbm)) install.packages("lightgbm", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(randomForest)
library(caret)
library(lubridate)
library(lightgbm)

# Set a seed
set.seed(1974)

# Load the final_holdout_test data set
final_holdout_test <- readRDS(rds_file_path_final)

# Create a random sample of maximum 500K records
max_records <- 500000
sampled_final_holdout_test <- final_holdout_test %>% sample_n(max_records)

# Save the sampled_final_holdout_test data frame to a new CSV file
write.csv(sampled_final_holdout_test, "sampled_final_holdout_test.csv", row.names = FALSE)

# Create training and test files in the 500-thousand-record file
# NOTE: test_set is set to 80%, and there is no model training

set.seed(1974)
indexes <- split(1:nrow(sampled_final_holdout_test), sampled_final_holdout_test$userId)
test_ind <- sapply(indexes, function(ind) sample(ind, ceiling(length(ind) * .8))) |>
  unlist(use.names = TRUE) |> sort()
test_set <- sampled_final_holdout_test[test_ind,]
train_set <- sampled_final_holdout_test[-test_ind,]

# Load library
library(dplyr)

# Compare column names
if (setequal(colnames(test_set), colnames(train_set))) {
  print("Column names of test_set and train_set are equal.")
} else {
  print("Column names of test_set and train_set are not equal.")
}

test_set <- test_set |> 
  semi_join(train_set, by = "movieId")
train_set <- train_set |> 
  semi_join(test_set, by = "movieId")

y <- select(train_set, movieId, userId, rating) |>
  pivot_wider(names_from = movieId, values_from = rating) 
rnames <- y$userId
y <- as.matrix(y[,-1])
rownames(y) <- rnames

movie_map <- train_set |> select(movieId, title) |> distinct(movieId, .keep_all = TRUE)

# Concatenate genres and remove '|'
sampled_final_holdout_test <- sampled_final_holdout_test %>%
  mutate(NewGenre = str_replace_all(genres, "\\|", ""))

colnames(sampled_final_holdout_test)

# Calculate the average rating per NewGenre
genre_avg_ratings <- sampled_final_holdout_test %>%
  group_by(NewGenre) %>%
  summarize(AvgRating = mean(rating, na.rm = TRUE))

count(genre_avg_ratings)

# Calculate the overall mean rating
mu <- mean(sampled_final_holdout_test$rating, na.rm = TRUE)

# Calculate the effect of each genre (b_g)
genre_effect <- genre_avg_ratings %>%
  mutate(b_g = AvgRating - mu) %>%
  select(NewGenre, b_g)

# Load libraries
library(ggplot2)
library(ggthemes)

# Create the plot of genre effect
plot_10 <- ggplot(genre_effect, aes(x = NewGenre, y = b_g)) +
  geom_point() +  # Scatter plot
  labs(x = "Number of Ratings", y = "Average Rating Effect") +
  theme_economist_white() +
  scale_x_discrete() +
  theme(axis.text.x = element_blank())  # Omit genre names on the x-axis

print(plot_10)

# Save plots for RDM file
plot_file_path_10 <- file.path(script_directory, "scatter_plot_avg_ratings_genre.png")
ggsave(plot_file_path_10, plot_10, width = 8, height = 6)

# Data preparation

# Load libraries
if (!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if (!require(lightgbm)) install.packages("lightgbm", repos = "http://cran.us.r-project.org")
library(lubridate)
library(lightgbm)

# Set a seed
set.seed(1974)

# Convert timestamp to datetime and extract day of the week
sampled_final_holdout_test$timestamp <- as.POSIXlt(sampled_final_holdout_test$timestamp, origin = "1970-01-01", tz = "UTC")
sampled_final_holdout_test$day_of_week <- weekdays(sampled_final_holdout_test$timestamp)

# Calculate mean rating for each day of the week
day_effect <- sampled_final_holdout_test %>%
  group_by(day_of_week) %>%
  summarize(mean_rating = mean(rating, na.rm = TRUE))

# Calculate the effect of the day of the week
day_effect$day_effect <- day_effect$mean_rating - mu

# Calculate movie effects
b_i <- colMeans(y - mu, na.rm = TRUE)

fit_movies <- data.frame(movieId = as.integer(colnames(y)), 
                         mu = mu, b_i = b_i)

# Add day_of_week to test_set
test_set$day_of_week <- weekdays(as.POSIXlt(test_set$timestamp, origin = "1970-01-01", tz = "UTC"))
train_set$day_of_week <- weekdays(as.POSIXlt(train_set$timestamp, origin = "1970-01-01", tz = "UTC"))

# Joining with day effect and movie effect
test_data <- left_join(test_set, fit_movies, by = "movieId") %>%
  left_join(day_effect, by = c("day_of_week" = "day_of_week"))

train_data <- left_join(train_set, fit_movies, by = "movieId") %>%
  left_join(day_effect, by = c("day_of_week" = "day_of_week"))

# Compare column names
colnames(train_data)
colnames(test_data)

# Add NewGenre column to train_set and test_set
train_set <- train_set %>%
  mutate(NewGenre = str_replace_all(genres, "\\|", ""))

test_set <- test_set %>%
  mutate(NewGenre = str_replace_all(genres, "\\|", ""))

# Compare column names
colnames(train_set)
colnames(test_set)

# Add genre_effect b_g to train_set and test_set
train_set <- left_join(train_set, genre_effect, by = "NewGenre")
test_set <- left_join(test_set, genre_effect, by = "NewGenre")

# Compare column names
colnames(train_set)
colnames(test_set)

# Joining with day effect and movie effect
test_data <- left_join(test_set, genre_effect, by = "NewGenre")
train_data <- left_join(train_set, genre_effect, by = "NewGenre")

# Compare column names
colnames(train_data)
colnames(test_data)

#v3
# Prepare data for LightGBM
lgb_data <- lgb.Dataset(data = as.matrix(test_data %>% select(-rating)), label = test_data$rating)

# Model prediction and performance

# Extract feature columns from test_data
test_features_v <- as.matrix(test_data %>% select(-rating))

# Make predictions using the LightGBM model
lgb_prediction_v <- predict(lgb_model, test_features_v)

# Calculate RMSE
rmse <- RMSE(lgb_prediction_v, test_data$rating)
print(rmse)
#RMSE: 0.9808728
#RMSE: 0.9699734
#RMSE: 0.9995787
#RMSE: 0.9813022 Sep 10, 2023 3:20 pm AM
# Code tested as of Sep 10, 2023 3:35 pm AM

# Code reviewed as of Sep 15, 2023 10:18 pm AM

#NEXT STEPS
# 1. Move last model to right sections OK Sep 10, 2023 4:09 pm AM
# 2. Plot and save plots for the model z OK Sep 12, 2023 9:28 pm AM
# 3. Review code OK Sep 15, 2023 10:22 pm AM
# 4. Edit RMD file OK Sep 22, 2023 10:28 pm AM
# 5. Produce RMD file OK Sep 22, 2023 10:30 pm AM
# 6. Upload to Github OK Sep 23, 2023 2:57 pm AM
# 7. Submit assignment. READY. Sep 23, 2023 6:14 pm AM
