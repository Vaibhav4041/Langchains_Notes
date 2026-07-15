from pydantic import BaseModel
from typing import Optional

class student(BaseModel):
    name: str
    age: Optional[int] = None # if not provided, defaults to None

new_student = {'name': 'Vaibhav Bhosale'}
new_student1 = {'name': 'Vaibhav Bhosale', 'age': 25}
new_student2 = {'name': 'Vaibhav Bhosale', 'age': '25'}
student_obj = student(**new_student)
student_obj1 = student(**new_student1)

print(student_obj2 := student(**new_student2))
print(student_obj1)
print(student_obj1.name)
print(student_obj)
print(student_obj.name)
print(type(student_obj))