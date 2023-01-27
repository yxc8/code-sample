import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.util.Scanner;
import java.io.PrintWriter;
import java.io.FileWriter;

public class cao_yiting_IP_Task5b {

    // Database credentials
    final static String HOSTNAME = "cao0016-sql-server.database.windows.net";
    final static String DBNAME = "cs-dsa-4513-sql-db";
    final static String USERNAME = "cao0016";
    final static String PASSWORD = "Xinkaishi20!";

    // Database connection string
    final static String URL = String.format("jdbc:sqlserver://%s:1433;database=%s;user=%s;password=%s;encrypt=true;trustServerCertificate=false;hostNameInCertificate=*.database.windows.net;loginTimeout=30;",
            HOSTNAME, DBNAME, USERNAME, PASSWORD);

    public static void main(String[] args) throws SQLException {
    	
    	System.out.println("WELCOME TO THE DATABASE SYSTEM OF MyProducts, Inc.");
		System.out.println("Enter an option to query, Eg. Option3");
		System.out.println("Append the required info after selected option with an asterisk in between, Eg. Option1*Jerry Sanders*36 Macro St, Santa Clara, CA*907652.67*worker*30* * * ");
		System.out.println("(1) Enter a new employee. Provide employee name, address, salary, employee type (worker/quality controller/technical staff), and type specific parameter(s) (worker: maximum number of product per day; quality controller: product type; technical staff: technical position, bachelor's degree, master's degree, phd degree [put blank space to that entry if none]).");
		System.out.println("(2) Enter a new product associated with the person who made the product, repaired the product if it is repaired, or checked the product. Provide product ID, date produced (8-digit, YMD), time spent producing (in minutes), worker name, quality controller name, technical staff name (put blank space if not repaired, Eg. * *), size of product, special attribute (For type 1 product, name of major software; for type 2 product, color of product; for type 3 product, weight of product with unit Eg. 30.1 lbs).  If repaired, also provide repair date (8 digits, YMD). If not repaired put 0.");
		System.out.println("(3) Enter a customer associated with some products. Provide customer name, address and product IDs as strings. Separate the IDs with \",\".");
		System.out.println("(4) Create a new account associated with a product. Provide account number, date account established (8-digit, YMD), cost of product, product category (1, 2, or 3) and productID.");
		System.out.println("(5) Enter a complaint associated with a customer and product. Provide customer name, product ID, complaint ID, complaint date, description and treatment expected.");
		System.out.println("(6) Enter an accident associated with an appropriate employee and product. Provide accident number, accident date, number of workdays lost, accident setting (during \"repair\" or \"produce\"), employee name, product ID.");
		System.out.println("(7) Retrieve the date produced and time spent to produce a particular product. Provide product ID.");
		System.out.println("(8) Retrieve all products made by a particular worker. Provide worker name.");
		System.out.println("(9) Retrieve the total number of errors a particular quality controller made. This is the total number of products certified by this controller and got some complaints. Provide quality controller name.");
		System.out.println("(10) Retrieve the total costs of the products in the product3 category which were repaired at the request of a particular quality controller. Provide quality controller name.");
		System.out.println("(11) Retrieve all customers (in name order) who purchased all products of a particular color. Provide color.");
		System.out.println("(12) Retrieve all employees whose salary is above a particular salary. Provide salary threshold.");
		System.out.println("(13) Retrieve the total number of workdays lost due to accidents in repairing the products which got complaints.");
		System.out.println("(14) Retrieve the average cost of all products made in a particular year. Provide year.");
		System.out.println("(15) Delete all accidents whose dates are in some range. Provide date lower threshold and date upper threshold.");
		System.out.println("(16) Import: enter new employees from a data file until the file is empty. Provide input file name.");
		System.out.println("(17) Export: Retrieve all customers (in name order) who purchased all products of a particular color and output them to a data file instead of screen. Provide color, export file name(container_name/file_name.extension).");
		System.out.println("(18) Quit");

    	// Take in command using scanner
    	Scanner myObj = new Scanner(System.in);
    	String command = myObj.nextLine();
    	while(!command.startsWith("Option18")) {
    		String[] tokens = command.split("\\*");
    		// Interpret user input
    		if (!command.startsWith("Option")) {
    			System.out.println("WRONG FORMAT! QUITTING...");
    			break;
    		}else {
    			int optionNumber = Integer.parseInt(tokens[0].substring(6));
    			connect: try (final Connection connection = DriverManager.getConnection(URL)) {
	    			if (optionNumber == 1) {
	    				if (tokens.length > 9 || tokens.length < 6) {
	    					System.out.println(tokens.length);
	    					System.out.println("WRONG NUMBER OF INPUT VALUES. TRY AGAIN.");
	    					break connect;
	    				}
	    	            try (final PreparedStatement statement = connection.prepareStatement("EXEC Option1 @ename = ?, @address = ?, @salary = ?, @type = ?, @param1 = ?, @param2a = ?, @param2b = ?, @param2c = ?;")) {
	    	            	double salary = Double.parseDouble(tokens[3]);
	    	            	
	    	            	// Setting the storage procedure input parameter values
	                        statement.setString(1, tokens[1]);
	                        statement.setString(2, tokens[2]);
	                        statement.setDouble(3, salary);
	                        statement.setString(4, tokens[4]);
	                        statement.setString(5, tokens[5]);
	                        statement.setString(6, tokens[6]);
	                        statement.setString(7, tokens[7]);
	                        statement.setString(8, tokens[8]);
		                    // Call the stored procedure
		                    statement.executeUpdate();
		
		                    System.out.println("Done.");
			            }
	    				
	    			}else if (optionNumber == 2) {
	    				if (tokens.length != 11) {
	    					System.out.println("WRONG NUMBER OF INPUT VALUES. TRY AGAIN.");
	    					break connect;
	    				}
	    	            try (final PreparedStatement statement = connection.prepareStatement("EXEC Option2 @prodID = ?, @dateProduced = ?, @timeSpent = ?, @wname = ?, @qcname = ?, @tsname = ?, @sizeOfProduct = ?, @productType = ?, @spAttr = ?, @repairDate = ?;")) {
	    	            	int prodID = Integer.parseInt(tokens[1]);
	    	            	int dateProduced = Integer.parseInt(tokens[2]);
	    	            	int timeSpent = Integer.parseInt(tokens[3]);
	    	            	int prodType = Integer.parseInt(tokens[8]);
	    	            	int repairDate = -1;
	    	            	if (tokens[6].length() != 0) {
	    	            		repairDate = Integer.parseInt(tokens[10]);
	    	            	}    
	    	            	
	    	            	// Setting the storage procedure input parameter values
	                        statement.setInt(1, prodID);
	                        statement.setInt(2, dateProduced);
	                        statement.setInt(3, timeSpent);
	                        statement.setString(4, tokens[4]);
	                        statement.setString(5, tokens[5]);
	                        statement.setString(6, tokens[6]);
	                        statement.setString(7, tokens[7]);
	                        statement.setInt(8, prodType);
	                        statement.setString(9, tokens[9]);
	                        // no need to prepare repairDate as it is an optional variable in t-sql
	                        if (tokens[6].length() != 0) {
	                        	statement.setInt(10, repairDate);
	    	            	}
	                        
	    				
		                    // Call the stored procedure
		                    statement.executeUpdate();
		                    
		                    System.out.println("Done.");

			            }
	    				
	    			}else if (optionNumber == 3) {
	    				if (tokens.length != 4) {
	    					System.out.println("WRONG NUMBER OF INPUT VALUES. TRY AGAIN.");
	    					break connect;
	    				}
	    	            try (final PreparedStatement statement = connection.prepareStatement("EXEC Option3 @cname = ?, @address = ?, @prodIDs = ?;")) {
	    	            	
	    	            	// Setting the storage procedure input parameter values
	                        statement.setString(1, tokens[1]);
	                        statement.setString(2, tokens[2]);
	                        statement.setString(3, tokens[3]);
	    				
		                    // Call the stored procedure
		                    statement.executeUpdate();
		
		                    System.out.println("Done.");
			            }
	    				
	    			}else if (optionNumber == 4) {
	    				if (tokens.length != 6) {
	    					System.out.println("WRONG NUMBER OF INPUT VALUES. TRY AGAIN.");
	    					break connect;
	    				}
	    	            try (final PreparedStatement statement = connection.prepareStatement("EXEC Option4 @anum = ?, @date = ?, @cost = ?, @productType = ?, @prodID = ?;")) {
	    	       
	    	            	int anum = Integer.parseInt(tokens[1]);
	    	            	int date = Integer.parseInt(tokens[2]);
	    	            	double cost = Double.parseDouble(tokens[3]);
	    	            	int prodType = Integer.parseInt(tokens[4]);
	    	            	int prodID = Integer.parseInt(tokens[5]);
	    	            		            	
	    	            	// Setting the storage procedure input parameter values
	                        statement.setInt(1, anum);
	                        statement.setInt(2, date);
	                        statement.setDouble(3, cost);
	                        statement.setInt(4, prodType);
	                        statement.setInt(5, prodID);
	    				
		                    // Call the stored procedure
		                    statement.executeUpdate();
		
		                    System.out.println("Done.");
			            }
	    				
	    			}else if (optionNumber == 5) {
	    				if (tokens.length != 7) {
	    					System.out.println("WRONG NUMBER OF INPUT VALUES. TRY AGAIN.");
	    					break connect;
	    				}
	    	            try (final PreparedStatement statement = connection.prepareStatement("EXEC Option5 @cname = ?, @prodID = ?, @complaintID = ?, @complaintDate = ?, @description = ?, @treatmentExpected = ?;")) {
	    	            	int prodID = Integer.parseInt(tokens[2]);
	    	            	int complaintID = Integer.parseInt(tokens[3]);
	    	            	int complaintDate = Integer.parseInt(tokens[4]);
	    	            	
	    	            	// Setting the storage procedure input parameter values
	                        statement.setString(1, tokens[1]);
	                        statement.setInt(2, prodID);
	                        statement.setInt(3, complaintID);
	                        statement.setInt(4, complaintDate);
	                        statement.setString(5, tokens[5]);
	                        statement.setString(6, tokens[6]);
	    				
		                    // Call the stored procedure
		                    statement.executeUpdate();
		
		                    System.out.println("Done.");
			            }
	    				
	    			}else if (optionNumber == 6) {
	    				if (tokens.length != 7) {
	    					System.out.println("WRONG NUMBER OF INPUT VALUES. TRY AGAIN.");
	    					break connect;
	    				}
	    	            try (final PreparedStatement statement = connection.prepareStatement("EXEC Option6 @anum = ?, @adate = ?, @numDaysLost = ?, @setting = ?, @ename = ?, @prodID = ?;")) {
	    	            	
	    	            	int anum = Integer.parseInt(tokens[1]);
	    	            	int adate = Integer.parseInt(tokens[2]);
	    	            	int numDaysLost = Integer.parseInt(tokens[3]);
	    	            	int prodID = Integer.parseInt(tokens[6]);

	    	            	
	    	            	// Setting the storage procedure input parameter values
	                        statement.setInt(1, anum);
	                        statement.setInt(2, adate);
	                        statement.setInt(3, numDaysLost);
	                        statement.setString(4, tokens[4]);
	                        statement.setString(5, tokens[5]);
	                        statement.setInt(6, prodID);
	    				
		                    // Call the stored procedure
		                    statement.executeUpdate();
		
		                    System.out.println("Done.");
			            }
	    				
	    			}else if (optionNumber == 7) {
	    				if (tokens.length != 2) {
	    					System.out.println("WRONG NUMBER OF INPUT VALUES. TRY AGAIN.");
	    					break connect;
	    				}
	    	            try (final PreparedStatement statement = connection.prepareStatement("EXEC Option7 @prodID = ?;")) {
	    	            	int prodID = Integer.parseInt(tokens[1]);
	    	            	
	    	            	// Setting the storage procedure input parameter values
	                        statement.setInt(1, prodID);
	                        
							// Call the stored procedure
							ResultSet resultSet = statement.executeQuery();
							  
							System.out.println("date produced | time spent ");
							  
							while (resultSet.next()) {
								System.out.println(String.format("%d | %d ",
							          resultSet.getInt(1),
							          resultSet.getInt(2)));
							}
	    			
		
		                    System.out.println("Done.");
			            }
	    				
	    			}else if (optionNumber == 8) {
	    				if (tokens.length != 2) {
	    					System.out.println("WRONG NUMBER OF INPUT VALUES. TRY AGAIN.");
	    					break connect;
	    				}
	    	            try (final PreparedStatement statement = connection.prepareStatement("EXEC Option8 @wname = ?;")) {	    	            	
	    	            	
	    	            	// Setting the storage procedure input parameter values
	                        statement.setString(1, tokens[1]);
	                        
//	                        output variables
//	                        product_ID INT PRIMARY KEY NONCLUSTERED,
//	                        date_produced INT,
//	                        time_spent_producing INT,
//	                        worker_name VARCHAR(50),
//	                        qc_name VARCHAR(50),
//	                        ts_name VARCHAR(50),
//	                        size_of_product VARCHAR(30),
	    				
		                    // Call the stored procedure
	                        ResultSet resultSet = statement.executeQuery();
	                        System.out.println("product ID | date produced | time spent producing | worker name | quality controller name | technical staff name | size of product ");
							  
							while (resultSet.next()) {
								System.out.println(String.format("%d | %d | %d | %s | %s | %s | %s ",
							          resultSet.getInt(1),
							          resultSet.getInt(2),
							          resultSet.getInt(3),
							          resultSet.getString(4),
							          resultSet.getString(5),
							          resultSet.getString(6),
							          resultSet.getString(7)));
							}
	                        
		
		                    System.out.println("Done.");
			            }
	    				
	    			}else if (optionNumber == 9) {
	    				if (tokens.length != 2) {
	    					System.out.println("WRONG NUMBER OF INPUT VALUES. TRY AGAIN.");
	    					break connect;
	    				}
	    	            try (final PreparedStatement statement = connection.prepareStatement("EXEC Option9 @qcname = ?;")) {
    	            	
	    	            	// Setting the storage procedure input parameter values
	    	            	statement.setString(1, tokens[1]);
	    				
	    	            	// Call the stored procedure
	                        ResultSet resultSet = statement.executeQuery();
	                        
	                        if (resultSet.next()) {
	                        	System.out.println(String.format("The total number of errors made by this quality controller is %d.", resultSet.getInt(1)));
	                        }else {
	                        	// No rows in the table
	                        	System.out.println("The total number of errors made by this quality controller is 0");
	                        }
	                        
		                    System.out.println("Done.");
			            }
	    				
	    			}else if (optionNumber == 10) {
	    				if (tokens.length != 2) {
	    					System.out.println("WRONG NUMBER OF INPUT VALUES. TRY AGAIN.");
	    					break connect;
	    				}
	    	            try (final PreparedStatement statement = connection.prepareStatement("EXEC Option10 @qcname = ?;")) {
	    	            	// Setting the storage procedure input parameter values
	    	            	statement.setString(1, tokens[1]);
	    				
	    	            	// Call the stored procedure
	                        ResultSet resultSet = statement.executeQuery();
	                        
	                        if(resultSet.next()) {
	                        	System.out.println(String.format("The total costs of the products in the product3 category which were repaired at the request of  this quality controller is %f.", resultSet.getDouble(1)));
		    	            }else {
	                        	// No rows in the table
	                        	System.out.println("No information on product3.");
	                        }
		
		                    System.out.println("Done.");
			            }
	    				
	    			}else if (optionNumber == 11) {
	    				if (tokens.length != 2) {
	    					System.out.println("WRONG NUMBER OF INPUT VALUES. TRY AGAIN.");
	    					break connect;
	    				}
	    	            try (final PreparedStatement statement = connection.prepareStatement("EXEC Option11 @color = ?;")) {
	    	            	// Setting the storage procedure input parameter values
	    	            	statement.setString(1, tokens[1]);
	    				
	    	            	// Call the stored procedure
	                        ResultSet resultSet = statement.executeQuery();
	                        System.out.println("customer name | customer address ");
							  
							while (resultSet.next()) {
								System.out.println(String.format("%s | %s ",
							          resultSet.getString(1),
							          resultSet.getString(2)));
							}
		
		                    System.out.println("Done.");
			            }
	    				
	    			}else if (optionNumber == 12) {
	    				if (tokens.length != 2) {
	    					System.out.println("WRONG NUMBER OF INPUT VALUES. TRY AGAIN.");
	    					break connect;
	    				}
	    	            try (final PreparedStatement statement = connection.prepareStatement("EXEC Option12 @salary = ?;")) {
	    	            	double salary = Double.parseDouble(tokens[1]);
	    	            	
	    	            	// Setting the storage procedure input parameter values
	                        statement.setDouble(1, salary);
	    				
	                     // Call the stored procedure
	                        ResultSet resultSet = statement.executeQuery();
	                        System.out.println("employee name | employee address | employee salary ");
	                        
							  
							while (resultSet.next()) {
								Double d = resultSet.getDouble(3);
								System.out.println(String.format("%s | %s | %f ",
							          resultSet.getString(1),
							          resultSet.getString(2),
							          d));
							}
		
		                    System.out.println("Done.");
			            }
	    			}else if (optionNumber == 13) {
	    				if (tokens.length != 1) {
	    					System.out.println("WRONG NUMBER OF INPUT VALUES. TRY AGAIN.");
	    					break connect;
	    				}
	    	            try (final PreparedStatement statement = connection.prepareStatement("EXEC Option13;")) {
	    				
	    	            	// Call the stored procedure
	                        ResultSet resultSet = statement.executeQuery();
	                        
	                        if(resultSet.next()) {
		                        System.out.println(String.format("The total number of workdays lost due to accidents in repairing the products which got complaints is %d.", resultSet.getInt(1)));
		    	            }else {
	                        	// No rows in the table
	                        	System.out.println("The total number of workdays lost due to accidents in repairing the products which got complaints is 0. Note there may be cases where complaints where made but products are not yet repaired.");
	                        }
		
		                    System.out.println("Done.");
			            }
	    				
	    			}else if (optionNumber == 14) {
	    				if (tokens.length != 2) {
	    					System.out.println("WRONG NUMBER OF INPUT VALUES. TRY AGAIN.");
	    					break connect;
	    				}
	    	            try (final PreparedStatement statement = connection.prepareStatement("EXEC Option14 @year = ?;")) {
	    	            	int year = Integer.parseInt(tokens[1]);
	    	            	
	    	            	// Setting the storage procedure input parameter values
	                        statement.setInt(1, year);
	    				
	                        // Call the stored procedure
	                        ResultSet resultSet = statement.executeQuery();
	                        
	                        if(resultSet.next()) {
	                        	Double d = resultSet.getDouble(1);
		                        System.out.println(String.format("The average cost of all products made in the provided year is %f.", d));
		    	            }else {
	                        	// No rows in the table
	                        	System.out.println("No information on this.");
	                        }
		
		                    System.out.println("Done.");
			            }
	    				
	    			}else if (optionNumber == 15) {
	    				if (tokens.length != 3) {
	    					System.out.println("WRONG NUMBER OF INPUT VALUES. TRY AGAIN.");
	    					break connect;
	    				}
	    	            try (final PreparedStatement statement = connection.prepareStatement("EXEC Option15 @ldate = ?, @udate = ?;")) {
	    	            	int ldate = Integer.parseInt(tokens[1]);
	    	            	int udate = Integer.parseInt(tokens[2]);
	    	            	
	    	            	// Setting the storage procedure input parameter values
	    	            	statement.setInt(1, ldate);
	    	            	statement.setInt(2, udate);
	    				
		                    // Call the stored procedure
		                    statement.executeUpdate();
		
		                    System.out.println("Done.");
			            }
	    				
					}else if (optionNumber == 16) {
						if (tokens.length != 2) {
	    					System.out.println("WRONG NUMBER OF INPUT VALUES. TRY AGAIN.");
	    					break connect;
	    				}
	    	            try (final PreparedStatement statement = connection.prepareStatement("EXEC Option16 @path = ?;")) {
	    	            	// Setting the storage procedure input parameter values
	    	            	statement.setString(1, tokens[1]);
	    				
		                    // Call the stored procedure
		                    statement.executeUpdate();
		
		                    System.out.println("Done.");
			            }
	    				
	    			}else if (optionNumber == 17) {
	    				if (tokens.length != 3) {
	    					System.out.println("WRONG NUMBER OF INPUT VALUES. TRY AGAIN.");
	    					break connect;
	    				}
	    	            try (final PreparedStatement statement = connection.prepareStatement("EXEC Option17 @color = ?;")) {

	    	            	// Setting the storage procedure input parameter values
	                        statement.setString(1, tokens[1]);

	                        
	                        // Call the stored procedure
	                        ResultSet resultSet = statement.executeQuery();
		                    
		                    try {	    	         
		                    	
	    	                    FileWriter fileWriter = new FileWriter(tokens[2]);
	    	                    PrintWriter printWriter = new PrintWriter(fileWriter);
	    	                    printWriter.print("name | address\n");
	    	                    while (resultSet.next()) {
	    	                    	printWriter.printf(String.format(" %s | %s \n", resultSet.getString(1), resultSet.getString(2)));
	    	                    }
	    	                    printWriter.close();
	    	                     	                    
	    	                 } catch(Exception e) {
	    	                    e.printStackTrace();
	    	                 }
		                    

		
		                    System.out.println("Result written to specified file.\n Done.");
			            }
	    				
	    			}else {
	    				System.out.println("NO SUCH OPTION. START AGAIN.");
	    				continue;
	    			}
	    		}
    		}
    		command = myObj.nextLine();
    	}
    	if (command.startsWith("Option18")) {
    		System.out.println("Quiting...");
			// Close scanner if asked to quit
    		myObj.close();
    		try (final Connection connection = DriverManager.getConnection(URL)) {
    			try (final PreparedStatement statement = connection.prepareStatement("EXEC Option18;")) {
    		
					// Call the stored procedure, https://docs.oracle.com/javase/7/docs/api/java/sql/Statement.html#executeUpdate%28java.lang.String%29
					// https://stackoverflow.com/questions/16348712/the-statement-did-not-return-a-result-set-java-error
					statement.executeUpdate();
    			}
			}

    		System.exit(0);
    	}
 
    }
}
