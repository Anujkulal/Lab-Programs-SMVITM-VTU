create database lab2;
use lab2;
create table emp(empno int primary key, ename varchar(50) not null, job varchar(50) not null, mgr int, sal decimal(10, 2) not null);
desc emp;

-- add a column commission
alter table emp add(commission decimal(10, 2));
desc emp;

-- insert any 5 records
insert into emp(empno, ename, job, mgr, sal, commission) values
(1, 'A', 'Manager', null, 50000.00, 1000.00),
(2, 'B', 'Assistant', 1, 35000.00, 500.00),
(3, 'C', 'clerk', 2, 25000.00, 500.00),
(4, 'D', 'clerk', 2, 25000.00, 500.00),
(5, 'E', 'clerk', 2, 25000.00, 500.00);
select * from emp;
 
-- update the column details of Job
update emp set job = "trainee" where empno = 5;
select * from emp;

-- rename the column of table 
alter table emp rename column mgr to ManagerNo;
alter table emp rename column sal to Salary;
select * from emp;

-- delete the employee whose empno is 5
delete from emp where empno = 5;
select * from emp;