package cao_yiting_IP_Task7;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.DriverManager;
import java.sql.PreparedStatement;

public class cao_yiting_IP_Task7_DataHandler {

    private Connection conn;

    // Azure SQL connection credentials
    private String server = "cao0016-sql-server.database.windows.net";
    private String database = "cs-dsa-4513-sql-db";
    private String username = "cao0016";
    private String password = "Xinkaishi20!";

    // Resulting connection string
    final private String url =
            String.format("jdbc:sqlserver://%s:1433;database=%s;user=%s;password=%s;encrypt=true;trustServerCertificate=false;hostNameInCertificate=*.database.windows.net;loginTimeout=30;",
                    server, database, username, password);

    // Initialize and save the database connection
    private void getDBConnection() throws SQLException {
        if (conn != null) {
            return;
        }

        this.conn = DriverManager.getConnection(url);
    }

    // Return the result of selecting everything from the movie_night table 
    public ResultSet getTopEmployees(Double salary) throws SQLException {
        getDBConnection();
        
        final PreparedStatement statement = conn.prepareStatement("EXEC Option12 @salary = ?;");
        	
    	// Setting the storage procedure input parameter values
        statement.setDouble(1, salary);
	
        // Call the stored procedure
        return statement.executeQuery();
        
    }

    // Inserts a record into the movie_night table with the given attribute values
    public boolean addEmployee(
            String ename, String address, String salary, String type, String p1, String p2a, String p2b, String p2c) throws SQLException {

        getDBConnection(); // Prepare the database connection
        
        try (final PreparedStatement statement = conn.prepareStatement("EXEC Option1 @ename = ?, @address = ?, @salary = ?, @type = ?, @param1 = ?, @param2a = ?, @param2b = ?, @param2c = ?;")) {
        	double s = Double.parseDouble(salary);
        	
        	// Setting the storage procedure input parameter values
            statement.setString(1, ename);
            statement.setString(2, address);
            statement.setDouble(3, s);
            statement.setString(4, type);
            statement.setString(5, p1);
            statement.setString(6, p2a);
            statement.setString(7, p2b);
            statement.setString(8, p2c);
            
            
		
            // Call the stored procedure
            return statement.executeUpdate() == 1;

        }
    }
}
