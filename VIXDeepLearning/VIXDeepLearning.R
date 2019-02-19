# clear variables and close windows
rm(list = ls(all = TRUE))
graphics.off()

# install and load packages
libraries = c("keras", "lubridate", "tidyverse", "reticulate")
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
    install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

# load the data Make surce the main repo folder has been set up as a working
# directory DATETIME is given in CET
df <- readRDS(file.path(getwd(), "VIXDeepLearning", "data", "raw", "VIX_Sp500.Rdata"))

# Calculate log-returns
log_df <- df %>% arrange(DATETIME) %>% group_by(DATE) %>% mutate_at(vars(-DATETIME, 
    -DATE), .funs = function(x) log(x/lag(x))) %>% ungroup() %>% select(DATETIME, 
    DATE, everything()) %>% mutate(VIX_PRED = lead(VIX)) %>% drop_na()

# Split the data into train and test datasets
train_interval <- interval("2018-01-01", "2018-01-31", tz = "CET")
train_df <- log_df %>% filter(DATETIME %within% train_interval)

x_train <- train_df %>% select(-DATE, -DATETIME, -VIX, -VIX_PRED)
y_train <- train_df %>% select(VIX_PRED)

test_interval <- interval("2018-02-01", "2018-02-28", tz = "CET")
test_df <- log_df %>% filter(DATETIME %within% test_interval)

x_test <- test_df %>% select(-DATE, -DATETIME, -VIX, -VIX_PRED)
y_test <- test_df %>% select(VIX_PRED)

# Data manipulation required by LSTM model
x_train_lstm <- as.matrix(x_train)
dim(x_train_lstm) <- c(dim(x_train)[[1]], 1, dim(x_train)[[2]])
y_train_lstm <- as.array(unlist(y_train))

x_test_lstm <- as.matrix(x_test)
dim(x_test_lstm) <- c(dim(x_test)[[1]], 1, dim(x_test)[[2]])
y_test_lstm <- as.array(unlist(y_test))

# Define the model
model <- keras_model_sequential()

model %>% layer_lstm(units = 50, input_shape = c(1, 10), batch_size = 10, return_sequences = FALSE, 
    activation = "tanh", stateful = TRUE, dropout = 0.2) %>% layer_dense(units = 1, 
    activation = "linear")

model %>% compile(loss = "mse", optimizer = optimizer_adam(lr = 10^-3, decay = 0.001), 
    metrics = c("mae", "mse"))

