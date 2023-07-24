def get_int_encoded(number):
  number *= 1000
  number = int(number)
  #print("n:",number)
  temp = number
  if(number%20 != 0):
    base_number = number//20
    base_number*=20
    if((number-base_number) <= 10 ):
			#return round(number/1000,2)
      temp = base_number
    else:
			#return round((number+20)/1000,2)
      temp = base_number+20
  temp = temp//10
  if temp>88:
    return 88
  if temp<-98:
    return -98
  return temp

l = [0.25, 0.025, -0.61, -0.061, 0.72, 0.072, 0.0064]
print()
for i in l:
    print(get_int_encoded(i))
