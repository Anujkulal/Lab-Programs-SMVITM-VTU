class Mypoint{
    private int x;
    private int y;

    public Mypoint(){
        this.x = 0;
        this.y = 0;
    }

    public Mypoint(int x, int y){
        this.x = x;
        this.y = y;
    }

    public void setXY(int x, int y){
        this.x = x;
        this.y = y;
    }

    public int[] getXY(){
        int[] coordinates = {x,y};
        return coordinates;
    }

    public double distance(int x, int y){
        int xdiff = this.x-x;
        int ydiff = this.y-y;
        return Math.sqrt(xdiff*xdiff + ydiff*ydiff);
    }

    public double distance(Mypoint another){
        int xdiff = this.x-another.x;
        int ydiff = this.y-another.y;
        return Math.sqrt(xdiff*xdiff + ydiff*ydiff);
    }

    public double distance(){
        return Math.sqrt(x*x + y*y);
    }

    @Override
    public String toString(){
        return "("+x+","+y+")";
    }
}

public class Testmypoint {
    public static void main(String[] args) {
        Mypoint point1 = new Mypoint();
        Mypoint point2 = new Mypoint(3, 4);
        point1.setXY(1, 2);
        int[] coordinates = point1.getXY();
        System.out.println("point 1 coordinates: ("+coordinates[0]+","+coordinates[1]+")");

        System.out.println("Distance from point1 to 5,6: "+point1.distance(5,6));
        System.out.println("distance from point1 to point2: "+point1.distance(point2));
        System.out.println("Distance from point1 to origin: "+point1.distance());

        System.out.println("point1: "+point1.toString());
        System.out.println("point2: "+point2.toString());
    }
}
