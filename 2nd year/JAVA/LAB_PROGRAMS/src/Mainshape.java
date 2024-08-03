class Shape{
    public void draw(){
        System.out.println("Drawing a shape");
    }
    public void erase(){
        System.out.println("Erasing a shape");
    }
}

class Circle extends Shape{
     public void draw(){
            System.out.println("Drawing a circle");
        }
        public void erase(){
            System.out.println("Erasing a circle");
        }
}

class Triangle extends Shape{
     public void draw(){
            System.out.println("Drawing a Triangle");
        }
        public void erase(){
            System.out.println("Erasing a triangle");
        }
}

class Square extends Shape{
     public void draw(){
            System.out.println("Drawing a square");
        }
        public void erase(){
            System.out.println("Erasing a square");
        }
}

public class Mainshape {
    public static void main(String[] args) {
        Shape[] shapes = new Shape[3];
        shapes[0] = new Circle();
        shapes[1] = new Triangle();
        shapes[2] = new Square();
        for(int i=0;i<shapes.length;i++){
            shapes[i].draw();
            shapes[i].erase();
            System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~");
        }
    }
}
