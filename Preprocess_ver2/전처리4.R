library(stringr)
library(tidyr)
library(readr)

setwd("C:/Users/jbk48/Desktop/민섭/철도연/download_webdisk.krri.re.kr(2019-04-14)/2018년 7월 19일/49편성")

north_data = read.csv("./전처리3/data_1_상행선.csv", sep=",")


code_number = parse_number(north_data$Next.Station.code)

new_dist = numeric()
new_dist[1] = north_data$Distance[1]

for(i in 2:nrow(north_data)){
  
  if(code_number[i-1] != code_number[i]){
    new_dist[i] = north_data$Distance[i]
  }else if(north_data$Distance[i]==0 && north_data$Distance[i-1] !=0){
    new_dist[i] = north_data$Distance[i]
  }
  else{
    new_dist[i] = north_data$Distance[i] - north_data$Distance[i-1]
  }
  
}

new_speed = c(0)

for(i in 2:nrow(north_data)){
  if(code_number[i] ==97 && code_number[i-1]==132){
    new_speed[i] = 0
  }else{
    new_speed[i] = north_data$Speed[i] - north_data$Speed[i-1]
  }
}

new_north_data = cbind(north_data, new_dist)

start_list = c(1)

for(i in 2:nrow(north_data)){
  
  if(code_number[i] == 97 && code_number[i-1] != 97 && code_number[i-1] == 132){
    start_list = append(start_list, i)
  }
}

end_list = numeric()

for(i in 1:(nrow(north_data)-1)){
  
  if(code_number[i] == 132 && code_number[i+1] != 132 && code_number[i+1] == 97){
    end_list = append(end_list, i)
  }
}

end_list = append(end_list, nrow(north_data))


remove_list = end_list - start_list

index_list = c()

for(i in 1: length(remove_list)){
  if(remove_list[i] > 4000 && remove_list[i] < 5000){
    index_list = append(index_list, i)
  }
}

ceiling(length(index_list)*0.75)
floor(length(index_list)*0.75)


train_index = index_list[1:floor(length(index_list)*0.75)]
test_index = index_list[ceiling(length(index_list)*0.75):length(index_list)]


train_df = data.frame()

for(index in train_index){
  
  abs_loc = cumsum(new_dist[start_list[index]:end_list[index]])
  accel = new_speed[start_list[index]:end_list[index]]
  sub_df = north_data[start_list[index]:end_list[index],]
  sub_df = cbind(sub_df, abs_loc)
  sub_df = cbind(sub_df, accel)
  train_df = rbind(train_df, sub_df)
}


write.csv(train_df, "./전처리3/data_1_north_train.csv", sep = "," , row.names = FALSE)


test_df = data.frame()

for(index in test_index){
  
  abs_loc = cumsum(new_dist[start_list[index]:end_list[index]])
  accel = new_speed[start_list[index]:end_list[index]]
  sub_df = north_data[start_list[index]:end_list[index],]
  sub_df = cbind(sub_df, abs_loc)
  sub_df = cbind(sub_df, accel)
  test_df = rbind(test_df, sub_df)
}


write.csv(test_df, "./전처리3/data_1_north_test.csv", sep = "," , row.names = FALSE)