create database lab5;
use lab5;
create table emp(eid int, ename varchar(50), age int, salary decimal(10, 2));
insert into emp(eid, ename, age, salary)values
(1, 'A', 30, 50000.00),
(2, 'B', 25, 45000.00),
(3, 'C', 35, 62000.00),
(4, 'D', 28, 52000.00),
(5, 'E', 32, 58000.00);
select * from emp;

-- create a stored procedure with cursor
delimiter //
create procedure fetch_emp_data()
begin
-- declare variable to store cursor values
declare emp_id int;
declare emp_name varchar(50);
declare emp_age int;
declare emp_salary decimal(10, 2);

declare emp_cursor cursor for -- declare a cursor for the table
select eid, ename, age, salary from emp;

declare continue handler for not found
set @finished = 1;

open emp_cursor;
set @finished = 0;

cursor_loop: loop
fetch emp_cursor into emp_id, emp_name, emp_age, emp_salary;
if @finished = 1 then leave cursor_loop;
end if;

select concat('Employee ID: ', emp_id, "Name: ", emp_name, "Age: ", emp_age, "Salary: ", emp_salary) as employee_info;
end loop;
close emp_cursor;
end; //
delimiter ;

call fetch_emp_data();