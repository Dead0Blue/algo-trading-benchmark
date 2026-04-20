package trading.data;

import tech.tablesaw.api.Table;
import tech.tablesaw.io.csv.CsvReadOptions;
import java.io.File;
import java.io.IOException;

public class DataFetcher {
    private static final String DATA_DIR = "../data";

    public Table loadData(String ticker) throws IOException {
        String filepath = DATA_DIR + File.separator + ticker + ".csv";
        System.out.println("Loading data for " + ticker + " from " + filepath);
        
        File file = new File(filepath);
        if (!file.exists()) {
            throw new IOException("File not found: " + filepath + ". Did you run the Python data fetcher first?");
        }
        
        CsvReadOptions options = CsvReadOptions.builder(file)
            .separator(',')
            .header(true)
            .build();
            
        Table table = Table.read().usingOptions(options);
        return table;
    }
}
