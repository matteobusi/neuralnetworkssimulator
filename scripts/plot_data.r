library(plotly)
library(ggplot2)
require(gridExtra)
library(reshape2)

setwd("~/Master/ML/Prj/NeuralNetworksSimulator")

plotGridRes3 <- function(basename, errname, index) {
    trainMSE <- data.frame(read.csv(paste("plots/cup/", basename, "_", errname, "_train_", index,".csv", sep=""), header=FALSE))
    valMSE <-  data.frame(read.csv(paste("plots/cup/", basename, "_", errname, "_val_", index,".csv", sep=""), header=FALSE))
    epochs <- 0:max(trainMSE$V2, na.rm = TRUE)

    fulldataset <- data.frame(epochs,
                             subset(trainMSE, V1 == 0, select=c(V2, V3)),
                             subset(trainMSE, V1 == 1, select=c(V2, V3)),
                             subset(trainMSE, V1 == 2, select=c(V2, V3)),
                             subset(valMSE, V1 == 0, select=c(V2, V3)),
                             subset(valMSE, V1 == 1, select=c(V2, V3)),
                             subset(valMSE, V1 == 2, select=c(V2, V3)))
    
    fulldataset <- subset(fulldataset, select=c(epochs, V3, V3.1, V3.2, V3.3, V3.4, V3.5))
    fulldataset$avgT <- apply(fulldataset[, 2:4], 1, mean)
    fulldataset$avgV <- apply(fulldataset[, 5:7], 1, mean)
    fulldataset$sdT <- apply(fulldataset[, 2:4], 1, sd)
    fulldataset$sdV <- apply(fulldataset[, 5:7], 1, sd)
    
    names(fulldataset) <- c("epoch", "f1T", "f2T", "f3T", "f1V",  "f2V", "f3V", "avgT", "avgV")

    ggplot() +
        geom_line(data=fulldataset, aes(x=epoch, y=f1T), color="red", linetype="solid", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=f2T), color="red", linetype="solid", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=f3T), color="red", linetype="solid", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=avgT), color="red", linetype="solid", size = 1, alpha=0.8) +
        geom_line(data=fulldataset, aes(x=epoch, y=f1V), color="blue", linetype="dashed", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=f2V), color="blue", linetype="dashed", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=f3V), color="blue", linetype="dashed", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=avgV), color="blue", linetype="dashed", size = 1, alpha=0.8) +
        xlab('# epoch') + ylab(toupper(errname))
}

plotGridRes5 <- function(basename, errname, index) {
    trainMSE <- data.frame(read.csv(paste("plots/cup/", basename, "_", errname, "_train_", index,".csv", sep=""), header=FALSE))
    valMSE <-  data.frame(read.csv(paste("plots/cup/", basename, "_", errname, "_val_", index,".csv", sep=""), header=FALSE))
    epochs <- 0:max(trainMSE$V2, na.rm = TRUE)
    
    fulldataset <- data.frame(epochs,
                              subset(trainMSE, V1 == 0, select=c(V2, V3)),
                              subset(trainMSE, V1 == 1, select=c(V2, V3)),
                              subset(trainMSE, V1 == 2, select=c(V2, V3)),
                              subset(trainMSE, V1 == 3, select=c(V2, V3)),
                              subset(trainMSE, V1 == 4, select=c(V2, V3)),
                              subset(valMSE, V1 == 0, select=c(V2, V3)),
                              subset(valMSE, V1 == 1, select=c(V2, V3)),
                              subset(valMSE, V1 == 2, select=c(V2, V3)),
                              subset(valMSE, V1 == 3, select=c(V2, V3)),
                              subset(valMSE, V1 == 4, select=c(V2, V3)))
    
    fulldataset <- subset(fulldataset, select=c(epochs, V3, V3.1, V3.2, V3.3, V3.4, V3.5, V3.6, V3.7, V3.8, V3.9))
    
    fulldataset$avgT <- apply(fulldataset[, 2:6], 1, mean)
    fulldataset$avgV <- apply(fulldataset[, 7:9], 1, mean)
    fulldataset$sdT <- apply(fulldataset[, 2:6], 1, sd)
    fulldataset$sdV <- apply(fulldataset[, 7:9], 1, sd)
    
    names(fulldataset) <- c("epoch", "f1T", "f2T", "f3T", "f4T", "f5T", "f1V",  "f2V", "f3V","f4V", "f5V", "avgT", "avgV")
    
    ggplot() +
        geom_line(data=fulldataset, aes(x=epoch, y=f1T), color="red", linetype="solid", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=f2T), color="red", linetype="solid", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=f3T), color="red", linetype="solid", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=f4T), color="red", linetype="solid", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=f5T), color="red", linetype="solid", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=avgT), color="red", linetype="solid", size = 1, alpha=0.8) +
        geom_line(data=fulldataset, aes(x=epoch, y=f1V), color="blue", linetype="dashed", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=f2V), color="blue", linetype="dashed", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=f3V), color="blue", linetype="dashed", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=f4V), color="blue", linetype="dashed", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=f5V), color="blue", linetype="dashed", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=avgV), color="blue", linetype="dashed", size = 1, alpha=0.8) +
        xlab('# epoch') + ylab(toupper(errname))
}

plotFinal <- function(basename, errname) {
    trainMSE <- data.frame(read.csv(paste("plots/cup/", basename, "_", errname, "_train.csv", sep=""), header=FALSE))
    valMSE <-  data.frame(read.csv(paste("plots/cup/", basename, "_", errname, "_val.csv", sep=""), header=FALSE))
    epochs <- 0:max(trainMSE$V2, na.rm = TRUE)
    
    fulldataset <- data.frame(epochs,
                              subset(trainMSE, V1 == 0, select=c(V2, V3)),
                              subset(trainMSE, V1 == 1, select=c(V2, V3)),
                              subset(trainMSE, V1 == 2, select=c(V2, V3)),
                              subset(trainMSE, V1 == 3, select=c(V2, V3)),
                              subset(trainMSE, V1 == 4, select=c(V2, V3)),
                              subset(valMSE, V1 == 0, select=c(V2, V3)),
                              subset(valMSE, V1 == 1, select=c(V2, V3)),
                              subset(valMSE, V1 == 2, select=c(V2, V3)),
                              subset(valMSE, V1 == 3, select=c(V2, V3)),
                              subset(valMSE, V1 == 4, select=c(V2, V3)))
    
    fulldataset <- subset(fulldataset, select=c(epochs, V3, V3.1, V3.2, V3.3, V3.4, V3.5, V3.6, V3.7, V3.8, V3.9))
    
    fulldataset$avgT <- apply(fulldataset[, 2:6], 1, mean)
    fulldataset$avgV <- apply(fulldataset[, 7:9], 1, mean)
    fulldataset$sdT <- apply(fulldataset[, 2:6], 1, sd)
    fulldataset$sdV <- apply(fulldataset[, 7:9], 1, sd)
    
    names(fulldataset) <- c("epoch", "f1T", "f2T", "f3T", "f4T", "f5T", "f1V",  "f2V", "f3V","f4V", "f5V", "avgT", "avgV")
    
    ggplot() +
        geom_line(data=fulldataset, aes(x=epoch, y=f1T), color="red", linetype="solid", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=f2T), color="red", linetype="solid", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=f3T), color="red", linetype="solid", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=f4T), color="red", linetype="solid", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=f5T), color="red", linetype="solid", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=avgT), color="red", linetype="solid", size = 1, alpha=0.8) +
        geom_line(data=fulldataset, aes(x=epoch, y=f1V), color="blue", linetype="dashed", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=f2V), color="blue", linetype="dashed", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=f3V), color="blue", linetype="dashed", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=f4V), color="blue", linetype="dashed", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=f5V), color="blue", linetype="dashed", size = 0.7, alpha=0.3) +
        geom_line(data=fulldataset, aes(x=epoch, y=avgV), color="blue", linetype="dashed", size = 1, alpha=0.8) +
        xlab('# epoch') + ylab(toupper(errname))
}

plotMonk <- function(basename, errname) {
    trainMSE <- data.frame(read.csv(paste(basename, "_", errname, "_train.csv", sep=""), header=FALSE))
    valMSE <-  data.frame(read.csv(paste(basename, "_", errname, "_test.csv", sep=""), header=FALSE))
    epochs <- 0:max(trainMSE$V2, na.rm = TRUE)
    
    fulldataset <- data.frame(epochs,
                              subset(trainMSE, V1 == 0, select=c(V2, V3)),
                              subset(valMSE, V1 == 0, select=c(V2, V3)))
    
    fulldataset <- subset(fulldataset, select=c(epochs, V3, V3.1))
    
    names(fulldataset) <- c("epoch", "f1Ts", "f1Te")
    
    ggplot() +
        geom_line(data=fulldataset, aes(x=epoch, y=f1Ts), color="red", linetype="solid", size = 1, alpha=0.8) +
        geom_line(data=fulldataset, aes(x=epoch, y=f1Te), color="blue", linetype="dashed", size = 1, alpha=0.8) +
        xlab('# epoch') + ylab(toupper(errname))
}

# Here we choose who to plot...

# Plot two examples for coarse...
k <- 7

p1 <- plotGridRes3("grid_coarse", "mse", k)
p2 <- plotGridRes3("grid_coarse", "mee", k)
pdf("/home/caos/Master/ML/report/plots/cup/coarse_8.pdf", width = 16, height = 8)
grid.arrange(p1, p2, ncol=2)
dev.off()
pz1 <- p1 + coord_cartesian(xlim = c(10,20),ylim= c(0,5))
pz2 <- p2 + coord_cartesian(xlim = c(10,20),ylim= c(0,5))
pdf("/home/caos/Master/ML/report/plots/cup/coarse_8_zoomed.pdf", width = 16, height = 8)
grid.arrange(pz1, pz2, ncol=2)
dev.off()
# 
k <- 16

p1 <- plotGridRes3("grid_coarse", "mse", k)
p2 <-  plotGridRes3("grid_coarse", "mee", k)

pdf("/home/caos/Master/ML/report/plots/cup/coarse_17.pdf", width = 16, height = 8)
grid.arrange(p1, p2, ncol=2)
dev.off()
# # Two examples for finer
k <- 7

p1 <- plotGridRes5("grid_finer", "mse", k)
p2 <-  plotGridRes5("grid_finer", "mee", k)

pdf("/home/caos/Master/ML/report/plots/cup/finer_8.pdf", width = 16, height = 8)
grid.arrange(p1, p2, ncol=2)
dev.off()
pz1 <- p1 + coord_cartesian(xlim = c(20,40),ylim= c(0,3))
pz2 <- p2 + coord_cartesian(xlim = c(20,40),ylim= c(0,3))

pdf("/home/caos/Master/ML/report/plots/cup/finer_8_zoomed.pdf", width = 16, height = 8)
grid.arrange(pz1, pz2, ncol=2)
dev.off()
# 
k <- 10

p1 <- plotGridRes5("grid_finer", "mse", k)
p2 <-  plotGridRes5("grid_finer", "mee", k)

pdf("/home/caos/Master/ML/report/plots/cup/finer_11.pdf", width = 16, height = 8)
grid.arrange(p1, p2, ncol=2)
dev.off()

# # And the final result on training/test
p1 <- plotFinal("final", "mse")
p2 <-  plotFinal("final", "mee")
pdf("/home/caos/Master/ML/report/plots/cup/final.pdf", width = 16, height = 8)
grid.arrange(p1, p2, ncol=2)
dev.off()

pz1 <- p1 + coord_cartesian(xlim = c(100,1000),ylim= c(0,5))
pz2 <- p2 + coord_cartesian(xlim = c(100,1000),ylim= c(0,5))

pdf("/home/caos/Master/ML/report/plots/cup/final_zoomed.pdf", width = 16, height = 8)
grid.arrange(pz1, pz2, ncol=2)
dev.off()

# Plot MONK's data

# ONLINE
p1 <- plotMonk("/home/caos/Master/ML/Prj/NeuralNetworksSimulator/plots/monk1/monk1_online", "mse")
p2 <- plotMonk("/home/caos/Master/ML/Prj/NeuralNetworksSimulator/plots/monk1/monk1_online", "acc")

pdf("/home/caos/Master/ML/report/plots/monk/monk1_online.pdf", width = 16, height = 8)
grid.arrange(p1, p2, ncol=2)
dev.off()

p1 <- plotMonk("/home/caos/Master/ML/Prj/NeuralNetworksSimulator/plots/monk2/monk2_online", "mse")
p2 <- plotMonk("/home/caos/Master/ML/Prj/NeuralNetworksSimulator/plots/monk2/monk2_online", "acc")

pdf("/home/caos/Master/ML/report/plots/monk/monk2_online.pdf", width = 16, height = 8)
grid.arrange(p1, p2, ncol=2)
dev.off()

p1 <- plotMonk("/home/caos/Master/ML/Prj/NeuralNetworksSimulator/plots/monk3/monk3_online", "mse")
p2 <- plotMonk("/home/caos/Master/ML/Prj/NeuralNetworksSimulator/plots/monk3/monk3_online", "acc")

pdf("/home/caos/Master/ML/report/plots/monk/monk3_online.pdf", width = 16, height = 8)
grid.arrange(p1, p2, ncol=2)
dev.off()


# BATCH (NO REG)
p1 <- plotMonk("/home/caos/Master/ML/Prj/NeuralNetworksSimulator/plots/monk1/monk1_batch_noreg", "mse")
p2 <- plotMonk("/home/caos/Master/ML/Prj/NeuralNetworksSimulator/plots/monk1/monk1_batch_noreg", "acc")

pdf("/home/caos/Master/ML/report/plots/monk/monk1_batch.pdf", width = 16, height = 8)
grid.arrange(p1, p2, ncol=2)
dev.off()

p1 <- plotMonk("/home/caos/Master/ML/Prj/NeuralNetworksSimulator/plots/monk2/monk2_batch_noreg", "mse")
p2 <- plotMonk("/home/caos/Master/ML/Prj/NeuralNetworksSimulator/plots/monk2/monk2_batch_noreg", "acc")

pdf("/home/caos/Master/ML/report/plots/monk/monk2_batch.pdf", width = 16, height = 8)
grid.arrange(p1, p2, ncol=2)
dev.off()

p1 <- plotMonk("/home/caos/Master/ML/Prj/NeuralNetworksSimulator/plots/monk3/monk3_batch_noreg", "mse")
p2 <- plotMonk("/home/caos/Master/ML/Prj/NeuralNetworksSimulator/plots/monk3/monk3_batch_noreg", "acc")

pdf("/home/caos/Master/ML/report/plots/monk/monk3_batch.pdf", width = 16, height = 8)
grid.arrange(p1, p2, ncol=2)
dev.off()

# BATCH (REG)
p1 <- plotMonk("/home/caos/Master/ML/Prj/NeuralNetworksSimulator/plots/monk3/monk3_batch_reg", "mse")
p2 <- plotMonk("/home/caos/Master/ML/Prj/NeuralNetworksSimulator/plots/monk3/monk3_batch_reg", "acc")

pdf("/home/caos/Master/ML/report/plots/monk/monk3_batchreg.pdf", width = 16, height = 8)
grid.arrange(p1, p2, ncol=2)
dev.off()



