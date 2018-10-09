library(ggplot2)
library(stringr)


r = read_file("./전처리2/list.txt")
filename = unlist(str_split(r, "\r\n"))[1:37]

north_data = data.frame()
south_data = data.frame()

for(name in filename){
    
    data = read.csv(paste("./전처리2/", name, sep = "") , sep = ",")
    
    sub_north_data = subset(data, data$head_north == 1)
    sub_south_data = subset(data, data$head_north == 0)
    
    north_data = rbind(north_data, sub_north_data)
    south_data = rbind(south_data, sub_south_data)
  
 }


write.csv(north_data, "./전처리3/data_1_상행선.csv" , sep ="," ,  row.names = FALSE)
write.csv(south_data, "./전처리3/data_1_하행선.csv" , sep ="," ,  row.names = FALSE)
nrow(south_data)
summary(north_data$Speed)
summary(south_data$Speed)

summary(north_data$Power)
summary(south_data$Power)

data$head_north = as.factor(data$head_north)

ggplot(data, aes(x=head_north, y=Speed, fill = head_north),width = 0.1) + geom_boxplot(width = 0.6) + 
  stat_summary(fun.y = "mean", geom="point", shape = 22, size = 5 , fill = "white") +
  scale_fill_discrete(name="Speed",breaks=c(0,1),labels=c("하행선","상행선"))

ggplot(data, aes(x=head_north, y=Power, fill = head_north),width = 0.1) + geom_boxplot(width = 0.6) + 
  stat_summary(fun.y = "mean", geom="point", shape = 22, size = 5 , fill = "white") +
  scale_fill_discrete(name="Power",breaks=c(0,1),labels=c("하행선","상행선"))