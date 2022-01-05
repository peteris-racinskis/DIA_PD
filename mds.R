#distances <- read.table("Results-1980-2015-dist.txt")
#names <- read.table("Results-1980-2015-name.txt")
distances <- read.table("classifier-distances.txt")
names <- read.table("classifier-names.txt")
library(MASS)
fit1 <- cmdscale(as.dist(distances), eig = TRUE, k = 2)
x <- fit1$points[, 1]
y <- fit1$points[, 2]
#plot(x,y, pch=4, col="grey",xlim=c(-3,4))
plot(x,y, pch=4, col="grey")
text(x,y,labels=names$V1)

fit2<-isoMDS(as.dist(distances), k=2)
x <- fit2$points[, 1]
y <- fit2$points[, 2]
#plot(x,y, pch=4, col="grey",xlim=c(-3,4))
plot(x,y, pch=4, col="grey")
text(x,y,labels=names$V1)
