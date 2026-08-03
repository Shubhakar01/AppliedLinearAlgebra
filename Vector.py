class Vector:
    """
    A Simple vector Class
    The Constructor expects a=n object that can be 
    correctly converted into list

    Try for Empty List
    """
    def __init__(self,values: list): #Type hint, not enforced
        self.values = list(values)

    def print_vector(self):
        print(self.values)

    def scalar_multiplication(self, scalar:float):
        #Always try to write in Compression
        #Multiplies the vector by a Scalar
        self.values = [scalar * x for x in self.values]

def test_vector_contents():
    v = Vector([1,2,3])
    #Used when Invarians Preconditions
    assert v.values == [1,2,3]

def test_empty_vector():
    v= Vector([])
    assert len(v.values)==0

def test_vector_ctor():
    v=Vector(())
    assert v.values == []

def test_vector_multiplication():
    v = Vector([1,2,3])
    v.scalar_multiplication(2)
    v.print_vector()

test_vector_multiplication()