# Python is an interpreted language, ulike Java, C and C++. The application does not need to be compiled into an executable to run it.
# It has its own Python interpreter, which is accessible through console.

def main():
    print ('Hello World!')
    
if __name__ == '__main__':
    main()
    
# The above chunk of code is used to run a function immediately when the file gets executed, usegful for the include modules 
    
# CONCAT ERROR: cannot concatenate a string and an int
print ("this is a string " + str(123))


#GLOBAL VS LOCAL
f=0

def someFunc():
    global f
    f="def"
    print(f)
    
someFunc()
print(f)