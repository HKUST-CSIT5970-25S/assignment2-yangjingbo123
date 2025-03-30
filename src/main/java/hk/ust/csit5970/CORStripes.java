package hk.ust.csit5970;

import org.apache.commons.cli.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;

/**
 * Compute the bigram correlation coefficients using "stripes" approach.
 */
public class CORStripes extends Configured implements Tool {
	private static final Logger LOG = Logger.getLogger(CORStripes.class);

	/*
	 * 第一阶段 Mapper：统计每个单词的总频数 Freq(A)
	 * 每行文本经过预处理后，按单词输出 (word,1)
	 */
	private static class CORMapper1 extends Mapper<LongWritable, Text, Text, IntWritable> {
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			// 使用规定的 tokenizer 对非字母字符进行过滤
			String cleanDoc = value.toString().replaceAll("[^a-z A-Z]", " ");
			StringTokenizer tokenizer = new StringTokenizer(cleanDoc);
			while (tokenizer.hasMoreTokens()) {
				String word = tokenizer.nextToken().toLowerCase();
				context.write(new Text(word), new IntWritable(1));
			}
		}
	}

	/*
	 * 第一阶段 Reducer：累加每个单词出现的次数，输出 (word, totalFrequency)
	 */
	private static class CORReducer1 extends Reducer<Text, IntWritable, Text, IntWritable> {
		@Override
		public void reduce(Text key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			context.write(key, new IntWritable(sum));
		}
	}

	/*
	 * 第二阶段 Mapper（Stripes 模式）：
	 * 每行文本先提取所有不重复单词（使用 TreeSet 保证排序），然后对于每个单词 A，
	 * 构造一个 stripe（MapWritable），记录该行中所有字典序在 A 后面的单词 B 出现一次。
	 * 输出键为 A，值为 stripe（其中 stripe 的 key 为 Text(B)，value 为 IntWritable(1)）。
	 */
	public static class CORStripesMapper2 extends Mapper<LongWritable, Text, Text, MapWritable> {
		@Override
		protected void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			// 提取不重复单词并排序
			Set<String> sortedSet = new TreeSet<>();
			String cleanDoc = value.toString().replaceAll("[^a-z A-Z]", " ");
			StringTokenizer tokenizer = new StringTokenizer(cleanDoc);
			while (tokenizer.hasMoreTokens()) {
				sortedSet.add(tokenizer.nextToken().toLowerCase());
			}
			// 将排序后的单词放入列表
			List<String> words = new ArrayList<>(sortedSet);
			// 对于每个单词 A，构造 stripe，统计后续单词 B（保证 A < B，避免重复计数）
			for (int i = 0; i < words.size(); i++) {
				String wordA = words.get(i);
				MapWritable stripe = new MapWritable();
				for (int j = i + 1; j < words.size(); j++) {
					Text wordB = new Text(words.get(j));
					// 如果该单词已在 stripe 中，则计数加 1，否则设为 1
					if (stripe.containsKey(wordB)) {
						IntWritable count = (IntWritable) stripe.get(wordB);
						count.set(count.get() + 1);
					} else {
						stripe.put(wordB, new IntWritable(1));
					}
				}
				if (!stripe.isEmpty()) {
					context.write(new Text(wordA), stripe);
				}
			}
		}
	}

	/*
	 * 第二阶段 Combiner（Stripes 模式）：
	 * 对 Mapper 发来的同一 key (word A) 的多个 stripe 进行合并，将同一单词 B 的计数相加。
	 */
	public static class CORStripesCombiner2 extends Reducer<Text, MapWritable, Text, MapWritable> {
		@Override
		protected void reduce(Text key, Iterable<MapWritable> values, Context context)
				throws IOException, InterruptedException {
			MapWritable combined = new MapWritable();
			for (MapWritable stripe : values) {
				for (MapWritable.Entry<Writable, Writable> entry : stripe.entrySet()) {
					Text wordB = (Text) entry.getKey();
					IntWritable count = (IntWritable) entry.getValue();
					if (combined.containsKey(wordB)) {
						IntWritable existing = (IntWritable) combined.get(wordB);
						existing.set(existing.get() + count.get());
					} else {
						combined.put(new Text(wordB), new IntWritable(count.get()));
					}
				}
			}
			context.write(key, combined);
		}
	}

	/*
	 * 第二阶段 Reducer（Stripes 模式）：
	 * 对每个 key (word A) 合并各个 stripe 得到最终 stripe，然后针对 stripe 中每个单词 B，
	 * 计算相关系数 COR(A, B) = Freq(A,B) / (Freq(A) * Freq(B))。
	 * 其中 Freq(A) 和 Freq(B) 从第一阶段中间文件中加载到 word_total_map 中。
	 * 输出键为 PairOfStrings(A, B)，值为计算得到的相关系数。
	 */
	public static class CORStripesReducer2 extends Reducer<Text, MapWritable, PairOfStrings, DoubleWritable> {
		private static Map<String, Integer> word_total_map = new HashMap<String, Integer>();

		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			Path middle_result_path = new Path("mid/part-r-00000");
			Configuration middle_conf = new Configuration();
			try {
				FileSystem fs = FileSystem.get(URI.create(middle_result_path.toString()), middle_conf);
				if (!fs.exists(middle_result_path)) {
					throw new IOException(middle_result_path.toString() + " not exist!");
				}
				FSDataInputStream in = fs.open(middle_result_path);
				InputStreamReader inStream = new InputStreamReader(in);
				BufferedReader reader = new BufferedReader(inStream);
				LOG.info("Reading middle result...");
				String line = reader.readLine();
				while (line != null) {
					String[] tokens = line.split("\t");
					if(tokens.length >= 2){
						word_total_map.put(tokens[0], Integer.valueOf(tokens[1]));
					}
					line = reader.readLine();
				}
				reader.close();
				LOG.info("Finished reading middle result.");
			} catch (Exception e) {
				System.out.println(e.getMessage());
			}
		}

		@Override
		protected void reduce(Text key, Iterable<MapWritable> values, Context context)
				throws IOException, InterruptedException {
			// 合并相同 key 的多个 stripe
			MapWritable combined = new MapWritable();
			for (MapWritable stripe : values) {
				for (MapWritable.Entry<Writable, Writable> entry : stripe.entrySet()) {
					Text wordB = (Text) entry.getKey();
					IntWritable count = (IntWritable) entry.getValue();
					if (combined.containsKey(wordB)) {
						IntWritable existing = (IntWritable) combined.get(wordB);
						existing.set(existing.get() + count.get());
					} else {
						combined.put(new Text(wordB), new IntWritable(count.get()));
					}
				}
			}
			// 获取当前 key 对应的总频数 Freq(A)
			Integer freqA = word_total_map.get(key.toString());
			if (freqA == null || freqA == 0) return;
			// 对 stripe 中的每个 B 计算 COR(A, B)
			for (MapWritable.Entry<Writable, Writable> entry : combined.entrySet()) {
				Text wordB = (Text) entry.getKey();
				Integer freqB = word_total_map.get(wordB.toString());
				if (freqB == null || freqB == 0) continue;
				IntWritable pairCountWritable = (IntWritable) entry.getValue();
				int pairCount = pairCountWritable.get();
				double corr = pairCount / (freqA.doubleValue() * freqB.doubleValue());
				// 输出要求中 A < B，由于 mapper 已保证 A 的 stripe中仅包含字典序在 A 后的单词 B，所以满足要求
				PairOfStrings pair = new PairOfStrings(key.toString(), wordB.toString());
				context.write(pair, new DoubleWritable(corr));
			}
		}
	}

	/**
	 * Creates an instance of this tool.
	 */
	public CORStripes() {
	}

	private static final String INPUT = "input";
	private static final String OUTPUT = "output";
	private static final String NUM_REDUCERS = "numReducers";

	/**
	 * Runs this tool.
	 */
	@SuppressWarnings({ "static-access" })
	public int run(String[] args) throws Exception {
		Options options = new Options();

		options.addOption(OptionBuilder.withArgName("path").hasArg()
				.withDescription("input path").create(INPUT));
		options.addOption(OptionBuilder.withArgName("path").hasArg()
				.withDescription("output path").create(OUTPUT));
		options.addOption(OptionBuilder.withArgName("num").hasArg()
				.withDescription("number of reducers").create(NUM_REDUCERS));

		CommandLine cmdline;
		CommandLineParser parser = new GnuParser();

		try {
			cmdline = parser.parse(options, args);
		} catch (ParseException exp) {
			System.err.println("Error parsing command line: " + exp.getMessage());
			return -1;
		}

		// Lack of arguments
		if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
			System.out.println("args: " + Arrays.toString(args));
			HelpFormatter formatter = new HelpFormatter();
			formatter.setWidth(120);
			formatter.printHelp(this.getClass().getName(), options);
			ToolRunner.printGenericCommandUsage(System.out);
			return -1;
		}

		String inputPath = cmdline.getOptionValue(INPUT);
		String middlePath = "mid";
		String outputPath = cmdline.getOptionValue(OUTPUT);

		int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

		LOG.info("Tool: " + CORStripes.class.getSimpleName());
		LOG.info(" - input path: " + inputPath);
		LOG.info(" - middle path: " + middlePath);
		LOG.info(" - output path: " + outputPath);
		LOG.info(" - number of reducers: " + reduceTasks);

		// Setup for the first-pass MapReduce (统计单词频数)
		Configuration conf1 = new Configuration();

		Job job1 = Job.getInstance(conf1, "Firstpass");
		job1.setJarByClass(CORStripes.class);
		job1.setMapperClass(CORMapper1.class);
		job1.setReducerClass(CORReducer1.class);
		job1.setOutputKeyClass(Text.class);
		job1.setOutputValueClass(IntWritable.class);

		FileInputFormat.setInputPaths(job1, new Path(inputPath));
		FileOutputFormat.setOutputPath(job1, new Path(middlePath));

		// 删除已存在的中间结果目录
		Path middleDir = new Path(middlePath);
		FileSystem.get(conf1).delete(middleDir, true);

		long startTime = System.currentTimeMillis();
		job1.waitForCompletion(true);
		LOG.info("Job 1 Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

		// Setup for the second-pass MapReduce (计算 COR 值)
		Path outputDir = new Path(outputPath);
		FileSystem.get(conf1).delete(outputDir, true);

		Configuration conf2 = new Configuration();
		Job job2 = Job.getInstance(conf2, "Secondpass");
		job2.setJarByClass(CORStripes.class);
		job2.setMapperClass(CORStripesMapper2.class);
		job2.setCombinerClass(CORStripesCombiner2.class);
		job2.setReducerClass(CORStripesReducer2.class);

		job2.setOutputKeyClass(PairOfStrings.class);
		job2.setOutputValueClass(DoubleWritable.class);
		job2.setMapOutputKeyClass(Text.class);
		job2.setMapOutputValueClass(MapWritable.class);
		job2.setNumReduceTasks(reduceTasks);

		FileInputFormat.setInputPaths(job2, new Path(inputPath));
		FileOutputFormat.setOutputPath(job2, new Path(outputPath));

		startTime = System.currentTimeMillis();
		job2.waitForCompletion(true);
		LOG.info("Job 2 Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

		return 0;
	}

	/**
	 * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
	 */
	public static void main(String[] args) throws Exception {
		ToolRunner.run(new CORStripes(), args);
	}
}

