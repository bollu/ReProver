#!/usr/bin/env python3
from typing import *
from tqdm import tqdm

class SineQuaNon:
    def __init__(self, axioms : Dict[str, str], symbols : List[str]):
        assert isinstance(axioms, dict)
        assert isinstance(symbols, list)
        self.symbols = symbols
        self.generality_threshold = 1
        self.axioms = axioms
        self._sym2numoccs = dict()
        self._axiom2syms = dict()
        self._sym2triggered = dict()
        # symbol -> number of axioms it occurs in
        # self._sym2numocc = self.build_occ()
        # axiom -> symbols that occur in axiom
        # self._axiom2syms = self.build_axiom2syms()
        # symbol -> axioms that this triggers.
        # This ordering is important. This can only be built once
        # the trigger relation is setup, which needs both
        # sym2numocc and axiom2syms to efficiently search for
        # whether the axiom is triggered by a symbol
        # self._sym2triggered = self.build_sym2triggered()


    def goal2selection(self, goal : str) -> List[List[str]]:
        print(f"goal2selection '{goal}'")
        frontier = set() # symbols to be visited.
        for sym in self.symbols:
            if self.occurs_in_goal(sym=sym, goal=goal):
                frontier.add(sym)

        seen = set()
        it = 0
        while len(frontier):
            axioms_per_level = []
            it += 1
            print(f"    iteration:'{it}' | trigger symbols: {str(frontier)}")
            new_frontier = set()
            for sym in tqdm(frontier, desc="    iterating on frontier..."):
                if sym in seen: continue
                seen.add(sym)
                for ax in self.sym2triggered(sym):
                    axioms_per_level.append(ax)
                    new_frontier = new_frontier.union(self.axiom2triggers(ax))
            print(f"    iteration:'{it}' | triggered axioms (#axioms={len(axioms_per_level)}): {str(axioms_per_level[:3])}")
            frontier = new_frontier
            yield axioms_per_level


    def is_triggered(self, sym : str, axiom : str):
        """
        Axiom is triggered by all symbols that have least occurrence.
        NOTE: there can be more than one trigger for a given axiom.
        """
        # sym is not in the axiom
        if not self.occurs_in_goal(sym=sym, goal=axiom): return False

        # any sufficiently specific premise always triggers
        if self.sym2numoccs(sym) < self.generality_threshold:
            return True

        for other_sym in self.axiom2syms(axiom):
            # this symbol is the trigger
            if self.sym2numoccs(other_sym) < self.sym2numoccs(sym):
                return False
        # Sym is in the axiom, and has the lowest occurrence. 
        # Thus, this symbol is in fact triggered.
        return True
        
    def sym2triggered(self, sym : str) -> List[str]:
        if sym in self._sym2triggered:
            return self._sym2triggered[sym]

        out = []
        for axiom_name in self.axioms:
            if self.is_triggered(sym=sym, axiom=self.axioms[axiom_name]):
                out.append(axiom_name)
        self._sym2triggered[sym] = out
        return out

    def axiom2syms(self, axiom: str):
        if axiom in self._axiom2syms:
            return self._axiom2syms[axiom]
        out = []
        for sym in self.symbols:
            if sym in axiom:
                out.append(sym)
        self._axiom2syms[axiom] = out
        return out

    def axiom2triggers(self, axiom : str):
        syms = self.axiom2syms(axiom)
        min_occ = min([self.sym2numoccs(sym) for sym in syms])
        out = [sym for sym in syms if self.sym2numoccs(sym) == min_occ]
        return out

    def sym2numoccs(self, sym : str) -> int:
        if sym in self._sym2numoccs:
            return self._sym2numoccs[sym]
        out = sum([self.occurs_in_goal(sym, ax) for ax in self.axioms])
        self._sym2numoccs[sym] = out
        return out


    def occurs_in_goal(self, sym, goal):
        """Return True if a occurs in b"""
        return sym in goal

if __name__ == "__main__":
    axioms = [
        "subclass(X, Y) /\ subclass(Y, Z) -> subclass(X, Z)",
        "subclass(petrol, liquid)",
        "not subclass(stone, liquid)",
        "subclass(beverage, liquid)",
        "subclass(beer, beverage)",
        "subclass(guiness, beer)",
        "subclass(pilsner, beer)",
    ]
    axioms = dict(list(zip(["name:" + ax[:20] + "..." for ax in axioms], axioms)))
    symbols = [
        "subclass",
        "liquid",
        "beer",
        "beverage",
        "petrol",
        "stone",
        "guiness",
        "pilsner"
    ]

    sqn = SineQuaNon(axioms, symbols)

    print(f"{'symbol':10} | {'occ'}")
    for symbol in symbols:
        print(f"{symbol:10} | {sqn.sym2numoccs(symbol):5}")
    print(f"{'='*80}")

    for axiom in axioms:
        triggers = []
        for sym in symbols:
            if sqn.is_triggered(sym=sym, axiom=axiom):
                triggers.append(sym)
        print(f"{axiom:50} | {' '.join(triggers):20}")

    print(f"{'='*80}")
    print(f"{'SYMBOL':50} | {'TRIGGERED AXIOMS'}")
    for sym in symbols:
        triggered = sqn.sym2triggered(sym)
        print(f"{sym:50} | {' '.join(triggered):20}")

    print(f"{'='*80}")
    print(f"{'GOAL':50} | {'USEFUL PREMISES'}")
    for axiom_name in axioms:
        print(f"{axioms[axiom_name]:50}")
        for (i, lvl) in enumerate(sqn.goal2selection(axioms[axiom_name])):
            print(f"    {i:4} | {str(lvl)}")

    print("="*80)
    GOAL2SELECTION_QUERY = "petrol => guiness"
    print(f"goal2selection({GOAL2SELECTION_QUERY})")
    for (i, lvl) in enumerate(sqn.goal2selection(GOAL2SELECTION_QUERY)):
        print(f"{i:4} | {str(lvl)}")
