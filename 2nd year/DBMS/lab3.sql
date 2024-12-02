create database lab3;
use lab3;

-- create emp table 
create table emp(eid int primary key, ename varchar(50) not null, age int not null, sal decimal(10, 2) not null);
desc emp;

-- count the no of employee names from the table
insert into emp(eid, ename, age, sal) values
(1, 'A', 20, 20000.00),
(2, 'B', 18, 50000.00),
(3, 'C', 21, 70000.00),
(4, 'D', 23, 30000.00),
(5, 'E', 20, 20000.00),
(6, 'F', 24, 40000.00);
select * from emp;
select count(ename) as Total_employees from emp;

-- Find the maximum age from table
select max(age) as Maximum_Age from emp;

-- Find the minimum age from table
select min(age) as Minimum_Age from emp;

-- find the salaries of employee in ascending order
select * from emp order by sal asc;

-- find the grouped salaries of employees
select sal, count(*) as Emp_Count from emp group by sal;