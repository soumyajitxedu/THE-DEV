public class practice_constutuer{
    String name;
    int exp;
    double duration;
    boolean ami;
    practice_constutuer(){
        name = "soumyajit";
        exp = 16;
        duration = 12.3;
        ami = true;


    }
    public static void main(String[] args){
        practice_constutuer oc = new practice_constutuer();
        System.out.println("name: "+ oc.name);
        System.out.println("exp is : "+ oc.exp);
        System.out.println("working duration or time is : "+ oc.duration);
        System.out.println("am i ? :"+oc.ami);
        
        
    }
    

}