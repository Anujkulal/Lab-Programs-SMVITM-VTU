class Employee{
    private int id;
    private String name;
    private double salary;

    public Employee(int id, String name, double salary){
        this.id=id;
        this.name=name;
        this.salary = salary;
    }

    public void raisesalary(double percent){
        if(percent>0){
            double raiseamount = salary * (percent/100);
            salary += raiseamount;
            System.out.println(name+"'s salary is raised by "+percent+". New salary: "+salary);
        }
        else{
            System.out.println("Invalid percentage, salary remains unchanged");
        }
    }

    public void employeedetails(){
        System.out.println("Employee ID: "+id+", Name: "+name+", Salary: $"+salary);
    }
}

public class Employee_sal {
    public static void main(String[] args) {
        Employee employee = new Employee(1, "Anuj", 50000);
        System.out.println("Initial employee details:");
        employee.employeedetails();

        employee.raisesalary(10);

        System.out.println("Employee details after salary raise:");
        employee.employeedetails();
    }
}
