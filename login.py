import json


def signin(email,password):
    with open('Data/Customers.json', 'r') as file:
        customers = json.load(file)
        if any(customer['email'] == email for customer in customers):
            if password == customer['password']:  # This is a placeholder; in a real application, you'd check hashed passwords
                return "Welcome back " + customer['name'] + "!"
        else:
            return "Invalid email or password."
def signup(name, email, password):
    with open('Data/Customers.json', 'r') as file:
        customers = json.load(file)
        if any(customer['email'] == email for customer in customers):
            return "Email already exists."
        else:
            new_customer = {"id": f"C{len(customers)+1:03}", "name": name, "email": email, "password": password}
            customers.append(new_customer)
            with open('Data/Customers.json', 'w') as file:
                json.dump(customers, file, indent=4)
            return "Account created successfully!"
    # Here you would typically check the email and password against a database
