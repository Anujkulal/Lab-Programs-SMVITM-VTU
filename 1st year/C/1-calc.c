/*simulation of a simple calculator*/
#include<stdio.h>
#include<stdlib.h>

void main()
{
float num1,num2,result;
char op;
printf("Enter the operators (+,-,*,/):");
scanf(" %c", &op);
printf("Enter the two numbers:");
scanf("%f%f", &num1,&num2);

if(op=='+'){
    result=num1+num2;
    printf("Result is %f",result);
}
else if(op=='-'){
    result=num1-num2;
    printf("Result is %f",result);
}
else if(op=='*'){
    result=num1*num2;
    printf("Result is %f",result);
}
else if(op=='/'){
    if(num2==0){
        printf("Enter the non-zero number!");
        exit(0);
    }
    else{
    result=num1/num2;
    printf("Result is %f",result);

    }
}
else{
    printf("Entered operator is invalid!!!");
}
}