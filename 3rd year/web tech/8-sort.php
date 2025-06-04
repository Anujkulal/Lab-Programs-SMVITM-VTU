<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>

    <style>
        body{
            font-family: 'Franklin Gothic Medium', 'Arial Narrow', Arial, sans-serif;
        }
        table{
            border-collapse: collapse;
            width: 100%;
        }
        th, td{
            border: 1px solid black;
        }
    </style>
</head>
<body>
    <h1>Sorted Student Records</h1>
    <table>
            <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Grade</th>
            </tr>

            <?php
            $conn = new mysqli("localhost", "root", "", "students");
            $students = $conn -> query("select * from students") -> fetch_all(MYSQLI_ASSOC);
            for($i=0; $i<count($students); $i++){
                $min = $i;
                for($j=$i+1; $j<count($students); $j++){
                    if($students[$j]['name'] < $students[$min]['name']) $min = $j;
                };
                $temp = $students[$min];
                $students[$min] = $students[$i];
                $students[$i] = $temp;
            }
            foreach($students as $student){
                echo " <tr>
                <th>{$student['id']}</th>
                <th>{$student['name']}</th>
                <th>{$student['grade']}</th>
            </tr>";
            };
            $conn -> close();
            ?>
    </table>
</body>
</html>