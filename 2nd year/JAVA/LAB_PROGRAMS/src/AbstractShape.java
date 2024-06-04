abstract class Shape1{
    abstract double calculatearea();
    abstract double calculateperimeter();
}

class Circle1 extends Shape1{
    private double radius;
    public Circle1(double radius){
        this.radius = radius;
    }
    @Override
    double calculatearea(){
        return Math.PI * radius*radius;
    }
    @Override
    double calculateperimeter(){
        return 2*Math.PI*radius;
    }
}

class Triangle1 extends Shape1{
    private double side1, side2, side3;
    public Triangle1(double side1, double side2, double side3){
        this.side1 = side1;
        this.side2 = side2;
        this.side3 = side3;
    }
    @Override
    double calculatearea(){
        double s = (side1+side2+side3)/2;
        return Math.sqrt(s*(s-side1)*(s-side2)*(s-side3));
    }

    double calculateperimeter(){
        return side1+side2+side3;
    }
}

public class AbstractShape {
    public static void main(String[] args) {
        Circle1 circle = new Circle1(5);
        System.out.println("Area of circle: "+circle.calculatearea());
        System.out.println("Perimeter of circle: "+circle.calculateperimeter());

        Triangle1 triangle  =new Triangle1(3,4,5);
        System.out.println("Area of triangle: "+triangle.calculatearea());
        System.out.println("Perimeter of triangle: "+triangle.calculateperimeter());
    }
}
