import argparse
parser = argparse.ArgumentParser(description='此代码用于整数求和')
#这个parse对象相当于一个总容器，存放着全部的信息
parser.add_argument("integers", nargs='*', help="所有整数")
#添加俩positional arguments，叫integers,读取类型为int
args = parser.parse_args()
#parse_args的类型是Namespaces，相当于一个字典，存放着所有positional arguments
integers=args.integers       #调出来intergers，它是一个list！！
print("integer的类型是:",type(integers))   
sum=sum(integers)  #list元素求和
print(sum)
