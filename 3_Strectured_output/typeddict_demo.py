from  typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    email: str

person1: Person = {
    'name': 'Alice',
    'age': 30,
    'email': 'xyz@gmail.com'
}   

print(person1)

