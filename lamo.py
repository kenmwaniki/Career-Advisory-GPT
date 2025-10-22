import json

people_string = """
{
  "people": [
    {
      "name": "John Smith",
      "emails": "null",
      "location": ["Kisii", "Nairobi", "Kenyenya"]
    },
    {
      "name": "Kegoro",
      "emails": ["kegoro@gmail.com", "wakis@gmail.com"],
      "location": "Nairobi"
    },
    {
      "name": "Kejonjo",
      "emails": ["nyakslamo@gmail.com", "Wales@gmail.com"],
      "location": "Mombasa"
    }
  ]
}
"""

data = json.loads(people_string)
for person in data['people']:
    print(person)

