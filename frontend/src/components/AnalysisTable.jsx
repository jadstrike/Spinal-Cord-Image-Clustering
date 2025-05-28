import React from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from "@mui/material";

export default function AnalysisTable({ data }) {
  if (!Array.isArray(data) || data.length === 0) return null;
  return (
    <TableContainer component={Paper} sx={{ maxWidth: 500 }}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Space</TableCell>
            <TableCell>Type</TableCell>
            <TableCell>Height</TableCell>
            <TableCell>Width</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {data.map((row, idx) => (
            <TableRow key={idx}>
              <TableCell>{row.Space}</TableCell>
              <TableCell>{row.Type}</TableCell>
              <TableCell>{row.Height}</TableCell>
              <TableCell>{row.Width}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
