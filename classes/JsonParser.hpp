#ifndef JSON_PARSER_HPP
#define JSON_PARSER_HPP

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

class JsonParser {
  public:
	// Write an object start
	static void writeObjectStart(std::ofstream &out, int indent = 0) {
		out << std::string(indent, ' ') << "{\n";
	}

	// Write an object end
	static void writeObjectEnd(std::ofstream &out, int indent = 0) {
		out << std::string(indent, ' ') << "}\n";
	}

	// Write an array start
	static void writeArrayStart(std::ofstream &out, int indent = 0) {
		out << std::string(indent, ' ') << "[\n";
	}

	// Write an array end
	static void writeArrayEnd(std::ofstream &out, int indent = 0) {
		out << std::string(indent, ' ') << "]\n";
	}

	// Write a key-value pair (for numbers)
	static void writeKeyValue(std::ofstream &out, const std::string &key,
							  float value, int indent = 0, bool comma = true) {
		out << std::string(indent, ' ') << "\"" << key << "\": " << value;
		if (comma) out << ",";
		out << "\n";
	}

	static void writeKeyValue(std::ofstream &out, const std::string &key,
							  size_t value, int indent = 0, bool comma = true) {
		out << std::string(indent, ' ') << "\"" << key << "\": " << value;
		if (comma) out << ",";
		out << "\n";
	}

	// Write a number to an array
	static void writeNumber(std::ofstream &out, float value,
							bool comma = true) {
		out << value;
		if (comma) out << ", ";
	}

	// Trim whitespace from string
	static void trim(std::string &s) {
		s.erase(s.begin(),
				std::find_if(s.begin(), s.end(),
							 [](unsigned char c) { return !std::isspace(c); }));
		s.erase(std::find_if(s.rbegin(), s.rend(),
							 [](unsigned char c) { return !std::isspace(c); })
					.base(),
				s.end());
	}

	// Validate numeric string
	static bool isValidFloat(const std::string &s) {
		if (s.empty()) return false;
		try {
			std::stof(s);
			return true;
		} catch (...) {
			return false;
		}
	}

	// Parse a float from content at position
	static float parseFloat(std::string &content, size_t &pos,
							const std::string &context = "") {
		while (pos < content.size() && std::isspace(content[pos])) ++pos;
		size_t start = pos;
		while (pos < content.size() && !std::isspace(content[pos]) &&
			   content[pos] != ',' && content[pos] != ']' &&
			   content[pos] != '}') {
			++pos;
		}
		std::string val = content.substr(start, pos - start);
		trim(val);
		if (val.empty() || !isValidFloat(val)) {
			throw std::runtime_error("Invalid float in " + context + ": '" +
									 val + "' at position " +
									 std::to_string(start));
		}
		return std::stof(val);
	}

	// Parse an unsigned integer from content at position
	static size_t parseSizeT(std::string &content, size_t &pos,
							 const std::string &context = "") {
		float value = parseFloat(content, pos, context);
		return static_cast<size_t>(
			std::stoul(std::to_string(static_cast<long>(value))));
	}

	// Skip to next expected character
	static void skipTo(std::string &content, size_t &pos, char target,
					   const std::string &context = "") {
		while (pos < content.size() && content[pos] != target) {
			++pos;
		}
		if (pos >= content.size()) {
			throw std::runtime_error("Expected character '" +
									 std::string(1, target) +
									 "' not found in " + context);
		}
		++pos;
	}

	// Read file content into string
	static std::string readFile(const std::string &filename) {
		std::ifstream in(filename);
		if (!in.is_open()) {
			throw std::runtime_error(
				"Unable to open file for deserialization: " + filename);
		}
		std::stringstream buffer;
		buffer << in.rdbuf();
		std::string content = buffer.str();
		in.close();
		return content;
	}
};

#endif
