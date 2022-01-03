dist.de <- read.table("Results-1980-2015-dist.txt")
names <- read.table("Results-1980-2015-name.txt")
fit <- cmdscale(dist.de, eig = TRUE, k = 2)
x <- fit$points[, 1]
y <- fit$points[, 2]
#plot(x,y, pch=4, col="grey",xlim=c(-2.5,-0.5))
plot(x,y, pch=4, col="grey")
text(x,y,labels=names$V1)
