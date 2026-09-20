from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name : str = 'purush'
    age : Optional[int]= None
    email: EmailStr
    cgpa: float = Field(gt=0,lt=10,default=5, description='A decimeal value representing the cgpa of the student')

# new_student = {}
new_student = {'age':22,'email':'abc@gmail.com'}

student = Student(**new_student)
# print(student);
student_dict = dict(student);
print(student_dict['age'])


# can use regex. for example in phone number field we can use regex to make it accept that 



