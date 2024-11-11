
#Practice Example
# Two erros in the code, use the debug to find.
#Make sure to set breakpoints to be able to use debugger efficently.

def sum_fact(x, y):
    resultx = 0
    resulty = 0
    
    # Perform some calculations and update values
    for i in range(x):
        resultx = x
        x -= 1
    
    for j in range(y):
        resulty = y
        y -= 1
    return resulty + resultx


final_result = sum_fact(3, 5)
print(f"Final result: {final_result}")
