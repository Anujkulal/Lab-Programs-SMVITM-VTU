create database lab1;
use lab1;


-- create a user and grant all permission to the user.
select user, host from mysql.user;
create user 'lab'@'%' identified by '1234';
grant all privileges on *.* to 'lab'@'%' with grant option;


-- insert any 3 records in the employee table and use rollback.
create table employee(empno int, ename varchar(50), job varchar(50), managerNo int, sal decimal(10, 2), commission decimal(10, 2));
desc employee;

insert into employee(empno, ename, job, managerNo, sal, commission)values
(1, 'A', 'Manager', null, 50000.00, 1000.00),
(2, 'B', 'Assistant', 1, 30000.00, 500.00),
(3, 'C', 'clerk', 2, 20000.00, 100.00);
select * from employee;

start transaction;
set autocommit = 0;
insert into employee(empno, ename, job, managerNo, sal, commission)values
(4, 'D', 'Trainee', null, 35000.00, 800.00);
select * from employee;

rollback;
select * from employee;


-- add primary key constraint and not null constraint to the table
alter table employee add primary key(empno);
desc employee;

alter table employee
modify ename varchar(50) not null,
modify job varchar(50) not null,
modify sal decimal(10, 2) not null;
desc employee;


-- insert the null values into the table and verify the result
insert into employee(empno, ename, job, managerNo, sal, commission)values
(4, 'D', 'Mary', 3, 35000.00, null);
select * from employee;

insert into employee(empno, ename, job, managerNo, sal, commission)values
(1, 'E', 'Manager', null, 50000.00, 1000.00); -- duplicate entry '1' for key 'employee.PRIMARY'

insert into employee(empno, ename, job, managerNo, sal, commission)values
(5, 'E', null, null, 50000.00, 1000.00); -- column 'job' cannot be null