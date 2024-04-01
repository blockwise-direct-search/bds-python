from rosenbrock_example import chrosen

if callable(chrosen):
    print("function_to_call is callable!")
else:
    print("function_to_call is not callable!")