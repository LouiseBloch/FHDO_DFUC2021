path_wd = "~/Workspace/dfuc2021/pseudo"
setwd(path_wd) # set working directory


# settings
####################################################################################################

# directory with pool of all images (train with unlabeled and test)
dir_images = "../images/all"

# directory to work in (here: id of ensemble configuration)
id = "ID6+ID7+ID13+ID19"

# sub-directory with files "probs_unlabeled.csv" and "probs_test.csv"
dir_probs = paste(id, "probs", sep = "/")

# paths to train labels and prediction probabilities for unlabeled and test images
path_train = "../train.csv"
path_probs_unlabeled = paste(dir_probs, "unlabeled_probs.csv", sep = "/")
path_probs_test = paste(dir_probs, "test_probs.csv", sep = "/")

# confidence threshold for predictions to consider as pseudo-labels
threshold = .7


# functions
####################################################################################################

# set labels based on prediction probabilities, filter less confident ones via threshold
# means: predictions with confidence lower threshold are removed, others are set to 1
threshmax <- function(df, threshold) {
  
  i_none = df$none >= threshold
  i_infection = df$infection >= threshold
  i_ischaemia = df$ischaemia >= threshold
  i_both = df$both >= threshold
  
  df$none = 0
  df$infection = 0
  df$ischaemia = 0
  df$both = 0
  
  df$none[i_none] = 1
  df$infection[i_infection] = 1
  df$ischaemia[i_ischaemia] = 1
  df$both[i_both] = 1
  
  df = subset(df, none == 1 | infection == 1 | ischaemia == 1 | both == 1)
  
  return(df)
}


# code
####################################################################################################

# load training labels and pseudo-labels (unlabeled, test)
data_train_labeled = na.omit(read.csv(path_train)) # train labels without unlabeled
data_probs_unlabeled = read.csv(path_probs_unlabeled)
data_probs_test = read.csv(path_probs_test)
cat("n unlabeled: ", length(data_probs_unlabeled[,1]))
cat("n test: ", length(data_probs_test[,1]))
cat("sum: ", length(data_probs_unlabeled[,1]) + length(data_probs_test[,1]))

# filter pseudo-labels by confidence and merge labels
data_probs_unlabeled_threshmax = threshmax(data_probs_unlabeled, threshold)
data_probs_test_threshmax = threshmax(data_probs_test, threshold)
cat("n unlabeled threshmax: ", length(data_probs_unlabeled_threshmax[,1]))
cat("n test threshmax: ", length(data_probs_test_threshmax[,1]))
cat("sum: ", length(data_probs_unlabeled_threshmax[,1]) + length(data_probs_test_threshmax[,1]))

data_pseudo = rbind(data_probs_unlabeled_threshmax, data_probs_test_threshmax)
data_pseudotrain = rbind(data_train_labeled, data_pseudo)
cat("none:", sum(data_pseudotrain$none), "with", sum(data_pseudo$none), "pseudolabels")
cat("infection:", sum(data_pseudotrain$infection), "with", sum(data_pseudo$infection), "pseudolabels")
cat("ischaemia:", sum(data_pseudotrain$ischaemia), "with", sum(data_pseudo$ischaemia), "pseudolabels")
cat("both:", sum(data_pseudotrain$both), "with", sum(data_pseudo$both), "pseudolabels")
cat("sum:", length(data_pseudotrain[,1]), "with", length(data_pseudo[,1]), "pseudolabels")

# split by class
class_pseudo_none = subset(data_pseudo, none == 1)
class_pseudo_ischaemia = subset(data_pseudo, ischaemia == 1)
class_pseudo_infection = subset(data_pseudo, infection == 1)
class_pseudo_both = subset(data_pseudo, both == 1)

class_pseudotrain_none = subset(data_pseudotrain, none == 1)
class_pseudotrain_ischaemia = subset(data_pseudotrain, ischaemia == 1)
class_pseudotrain_infection = subset(data_pseudotrain, infection == 1)
class_pseudotrain_both = subset(data_pseudotrain, both == 1)


# create file system structure
dir_split = paste(id, "split", sep = "/")
dir_none = paste(dir_split, "none", sep = "/")
dir_ischaemia = paste(dir_split, "ischaemia", sep = "/")
dir_infection = paste(dir_split, "infection", sep = "/")
dir_both = paste(dir_split, "both", sep = "/")

dir.create(dir_split)
dir.create(dir_none)
dir.create(dir_ischaemia)
dir.create(dir_infection)
dir.create(dir_both)


# copy files from image pool in split dirs, write separate label files
file.copy(paste(dir_images, class_pseudotrain_none$image, sep = "/"), dir_none)
file.copy(paste(dir_images, class_pseudotrain_ischaemia$image, sep = "/"), dir_ischaemia)
file.copy(paste(dir_images, class_pseudotrain_infection$image, sep = "/"), dir_infection)
file.copy(paste(dir_images, class_pseudotrain_both$image, sep = "/"), dir_both)

write.csv(data_pseudotrain, paste(dir_split, "all.csv", sep = "/"), quote = F, row.names = F)
write.csv(class_pseudotrain_none, paste(dir_split, "none.csv", sep = "/"), quote = F, row.names = F)
write.csv(class_pseudotrain_ischaemia, paste(dir_split, "ischaemia.csv", sep = "/"), quote = F, row.names = F)
write.csv(class_pseudotrain_infection, paste(dir_split, "infection.csv", sep = "/"), quote = F, row.names = F)
write.csv(class_pseudotrain_both, paste(dir_split, "both.csv", sep = "/"), quote = F, row.names = F)
