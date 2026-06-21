function helper(x: number): number {
  return x + 1;
}

function caller(items: number[]): number[] {
  return items.map((x) => helper(x));
}
