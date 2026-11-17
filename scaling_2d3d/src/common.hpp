// common.hpp -- tiny command-line parser shared by all executables.
// Style: "--key value" or "--key=value" or bare "--flag" (= "1").
// Every key that is read is recorded; unknown/misspelled keys are reported by unused().
#pragma once
#include <cstdlib>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

class Args {
 public:
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string s = argv[i];
      if (s.rfind("--", 0) != 0) continue;
      s = s.substr(2);
      size_t eq = s.find('=');
      if (eq != std::string::npos) {
        kv_[s.substr(0, eq)] = s.substr(eq + 1);
      } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        kv_[s] = argv[++i];
      } else {
        kv_[s] = "1";
      }
    }
  }
  bool has(const std::string& k) const { return kv_.count(k) > 0; }
  std::string str(const std::string& k, const std::string& def) const {
    used_.insert(k);
    auto it = kv_.find(k);
    return it == kv_.end() ? def : it->second;
  }
  int integer(const std::string& k, int def) const {
    used_.insert(k);
    auto it = kv_.find(k);
    return it == kv_.end() ? def : std::atoi(it->second.c_str());
  }
  double real(const std::string& k, double def) const {
    used_.insert(k);
    auto it = kv_.find(k);
    return it == kv_.end() ? def : std::atof(it->second.c_str());
  }
  std::vector<int> int_list(const std::string& k, const std::vector<int>& def) const {
    used_.insert(k);
    auto it = kv_.find(k);
    if (it == kv_.end()) return def;
    std::vector<int> out;
    std::stringstream ss(it->second);
    std::string tok;
    while (std::getline(ss, tok, ',')) out.push_back(std::atoi(tok.c_str()));
    return out;
  }
  std::vector<std::string> unused() const {
    std::vector<std::string> u;
    for (auto& p : kv_)
      if (!used_.count(p.first)) u.push_back(p.first);
    return u;
  }

 private:
  std::map<std::string, std::string> kv_;
  mutable std::set<std::string> used_;
};
