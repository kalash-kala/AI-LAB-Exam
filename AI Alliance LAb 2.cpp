#include<bits/stdc++.h>

using namespace std;

class Node {
public:
    Node* next;
    int val;
    Node(int val) {
        this -> next = nullptr;
        this -> val = val;
    }
};

// 0 - 1
class Random {
    vector<float> randoms;
    public:
    Random() {
        srand(time(0));
        srand((unsigned)time(NULL));
        for(int i = 0; i < 10000000; i++)
            randoms.push_back((float)rand()/RAND_MAX), randoms.back();
    }
    vector<float> getRandoms() {
        return randoms;
    }
    
};

class State {
public:
    Node *s1, *s2, *s3, *s4, *s5, *s6, *s7, *s8, *s9, *s10;
    State() {
        s1 = new Node(100);
        s2 = new Node(500);
        s3 = new Node(1000);
        s4 = new Node(5000);
        s5 = new Node(10000);
        s6 = new Node(50000);
        s7 = new Node(100000);
        s8 = new Node(500000);
        s9 = new Node(1000000);
        s10 = new Node(5000000);
        s1 -> next = s2;
        s2 -> next = s3;
        s3 -> next = s4;
        s4 -> next = s5;
        s5 -> next = s6;
        s6 -> next = s7;
        s7 -> next = s8;
        s8 -> next = s9;
        s9 -> next = s10;
    }
};

class iterate {
    bool stopper = false;
    map<int, int> TimesVisited;
    map<int, double> ValueFunction;
    Node* head;
    double gamma = 1, theta = 0.00005;
    vector<float> prob = {1, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0},  randoms;
    int itr = 0;
public:
    int cnt = 0;
    iterate() {
        State obj;
        this -> head = obj.s1;
        Random obj1;
        this -> randoms = obj1.getRandoms();
    }

    void PrintStateTransition() {
        for(auto it: TimesVisited)
            cout << it.first << " ";
        cout << "\n";
        for(auto it: TimesVisited)
            cout << ((double)it.second * 100) / cnt << " ";
        cout << "\n";
    }

    bool getStopperState() {
        return stopper;
    }

    pair<int, double> dfs(Node* temp, int currState) {
        TimesVisited[currState]++;
        if(!temp)
            return {currState, 0};
        int maxStateReached = currState;
        double answerReward = 0, mx = 0, oldValue = ValueFunction[currState], quitReward = ValueFunction[currState - 1];
        pair<int, double> nextStateReward;
        if(randoms[itr++ % 10000000] <= prob[currState]) {
            nextStateReward = dfs(temp->next, currState + 1), answerReward = prob[currState] * (gamma * nextStateReward.second + temp->val), maxStateReached = max(maxStateReached, nextStateReward.first), ValueFunction[currState] = (ValueFunction[currState] * (TimesVisited[currState] - 1) + answerReward)/ (TimesVisited[currState]);
            if(abs(oldValue - ValueFunction[currState])    < theta)
                stopper = true;
        }
        return {maxStateReached, max(quitReward, answerReward)};
    }

    pair<int, double> cycle() {
        cnt++;
        return dfs(head, 1);
    }
    
};

class Solution {
    int itr = 0;
    unordered_map<int, int> mp;
    unordered_map<int, double> mp1;
public:
    void solve() {
        iterate obj;
        while(!obj.getStopperState()){
            auto it = obj.cycle();
            mp[it.first]++, itr++;
            mp1[it.first] = (mp1[it.first] * (obj.cnt - 1) + it.second) / obj.cnt;
        }
        // Prbability of ending up at this state.
        cout << itr << '\n';
        obj.PrintStateTransition();
    }


    void print() {
        vector<pair<int, int>> pr, pr1;
        for(auto it : mp)
            pr.push_back(it);
        for(auto it : mp1)
            pr1.push_back(it);
        sort(pr1.begin(), pr1.end());
        sort(pr.begin(), pr.end());
        for(auto it: pr)
            cout << it.first << " ";
        cout << "\n";
        for(auto it: pr)
            cout << ((double)it.second * 100) / itr << " ";
        cout << "\n";
        for(auto it: pr1)
            cout << it.first << " ";
        cout << "\n";
        for(auto it: pr1)
            cout << it.second << " ";
        cout << "\n";
    }
};

// Driver code
int main() {
    Solution obj;
    obj.solve();
    obj.print();
    return 0;
}
